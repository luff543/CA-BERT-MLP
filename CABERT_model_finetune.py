# -*- coding: utf-8 -*-
# @Time : 2022/9/26 19:38
# @Author : luff543
# @Email : luff543@gmail.com
# @File : CABERT_model_finetune.py
# @Software: PyCharm

import os
import numpy as np
from BERT_multi_label_utils import *
from bert import optimization

global ids_array
global mask_array
global segment_array

ids_array = []
mask_array = []
segment_array = []

BERT_MODULE = None


class CABERTModel(object):

    def __init__(self, config):
        self.model_config = config
        self.logger = config["logger"]

        batch_size = config["batch_size"]
        num_labels = config["num_labels"]
        sentence_fragment_length = config["sentence_fragment_length"]

        num_gru_layers = config["num_gru_layers"]
        gru_hidden_size = config["gru_hidden_size"]
        l2_lambda = config["l2_lambda"]

        if "is_training" in config:
            self.is_training = config["is_training"]
        else:
            self.is_training = None

        if self.is_training == True:
            num_train_steps = config["num_train_steps"]
            num_warmup_steps = config["num_warmup_steps"]

        self.input_ids = config["input_ids"]
        self.input_mask = config["input_mask"]
        self.segment_ids = config["segment_ids"]
        self.label_ids = config["label_ids"]

        print_shape('input_ids', self.input_ids)
        print_shape('input_mask', self.input_mask)
        print_shape('segment_ids', self.segment_ids)
        print_shape('label_ids', self.label_ids)

        """
        input_ids: (?, 10, 128)
        input_mask: (?, 10, 128)
        segment_ids: (?, 10, 128)
        label_ids: (?,)
        """

        self.initializers = config["initializers"]

        global ids_array
        global mask_array
        global segment_array

        ids_array = tf.split(self.input_ids, num_or_size_splits=sentence_fragment_length, axis=1)
        mask_array = tf.split(self.input_mask, num_or_size_splits=sentence_fragment_length, axis=1)
        segment_array = tf.split(self.segment_ids, num_or_size_splits=sentence_fragment_length, axis=1)

        output = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        step = tf.constant(0)
        _, seq_outputs = tf.while_loop(self.cond, self.body, loop_vars=[step, output])
        seq_outputs = seq_outputs.stack()
        tf.logging.info("original seq_outputs: {}".format(seq_outputs))
        seq_outputs = tf.transpose(seq_outputs, perm=[1, 0, 2])
        tf.logging.info("seq_outputs: {}".format(seq_outputs))

        seq_outputs_position_embedding = self._positional_encoding_offset(seq_outputs, maxlen=sentence_fragment_length,
                                                                          masking=False, scope="positional_encoding",
                                                                          offset_position=0)
        seq_outputs_add_seq_outputs_position_embedding = seq_outputs + seq_outputs_position_embedding

        self.seq_outputs_position_embedding = seq_outputs_position_embedding
        self.seq_outputs_add_seq_outputs_position_embedding = seq_outputs_add_seq_outputs_position_embedding

        gru_output = self._multi_label_classifier_bigru_output(
            seq_outputs_add_seq_outputs_position_embedding, 'GRU', num_gru_layers,
            gru_hidden_size)

        tf.logging.info("num_labels: {};gru_output: {};label_ids: {}".format(num_labels, gru_output, self.label_ids))
        # project
        logits = self.project_bi_gru_layer(gru_output)

        tf.logging.info("num_labels: {};logits: {};label_ids: {}".format(num_labels, logits, self.label_ids))

        loss = self._loss_op(l2_lambda, self.label_ids, logits)
        _, predicted_labels_flat, probabilities_flat, probabilities = self._predict_op(self.label_ids, logits)
        tf.logging.info("num_labels: {};predicted_labels_flat".format(num_labels, predicted_labels_flat))
        labels_flat = tf.reshape(self.label_ids, [batch_size * sentence_fragment_length, num_labels])
        logits_flat = tf.reshape(logits, [batch_size * sentence_fragment_length, num_labels])
        tf.logging.info("num_labels: {};labels_flat: {};logits_flat: {}".format(num_labels, labels_flat, logits_flat))

        self.labels = self.label_ids
        self.logits = logits
        self.labels_flat = labels_flat
        self.logits_flat = logits_flat
        self.loss = loss
        self.global_step = step

        self.probabilities = probabilities
        self.probabilities_flat = probabilities_flat

        learning_rate = 5e-5

        tf.summary.scalar('train loss', loss)
        if self.is_training:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            self.train_op = train_op
        self.init_global_op = tf.global_variables_initializer()
        self.init_local_op = tf.initialize_local_variables()

    def _multi_label_classifier_bigru_output(self, inputs, scope, num_gru_layers, num_units):
        dropout_keep_prob = self.model_config["dropout_keep_prob"]

        with tf.variable_scope(scope):
            gru_fw_cells = []
            for _ in range(num_gru_layers):
                gru_fw_cell = tf.contrib.rnn.GRUCell(num_units)
                gru_fw_cell = tf.contrib.rnn.DropoutWrapper(
                    gru_fw_cell, output_keep_prob=dropout_keep_prob)
                gru_fw_cells.append(gru_fw_cell)
            gru_forward = tf.nn.rnn_cell.MultiRNNCell(gru_fw_cells)

            gru_bc_cells = []
            for _ in range(num_gru_layers):
                gru_bc_cell = tf.contrib.rnn.GRUCell(num_units)
                gru_bc_cell = tf.contrib.rnn.DropoutWrapper(
                    gru_bc_cell, output_keep_prob=dropout_keep_prob)
                gru_bc_cells.append(gru_bc_cell)
            gru_backward = tf.nn.rnn_cell.MultiRNNCell(gru_bc_cells)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_forward, cell_bw=gru_backward,
                                                                        inputs=inputs, dtype=tf.float32,
                                                                        time_major=False)
            print_shape('(_bi_gru) output_fw', output_fw)
            print_shape('(_bi_gru) output_bw', output_bw)
            output = tf.concat([output_fw, output_bw], axis=-1)  # axis=-1 Represents the penultimate first dimension

            self.output_fw = output_fw
            self.output_bw = output_bw
            self.output = output
            # output = tf.concat([output_fw, output_bw], axis=2)

            print_shape('(_bi_gru) output', output)
            # output = tf.reshape(output, (-1, sentence_fragment_length, num_units * 2))
            print_shape('(_bi_gru) output', output)

            return output

    def project_bi_gru_layer(self, gru_outputs, name=None):
        """
        hidden layer between gru layer and logits
        :param gru_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_labels]
        """

        num_labels = self.model_config["num_labels"]
        gru_hidden_size = self.model_config["gru_hidden_size"]
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[gru_hidden_size * 2, gru_hidden_size],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[gru_hidden_size], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(gru_outputs, shape=[-1, gru_hidden_size * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[gru_hidden_size, num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.model_config["sentence_fragment_length"], num_labels])

    # calculate classification loss
    def _loss_op(self, l2_lambda, y, logits):
        y = tf.cast(y, tf.float32)
        with tf.name_scope('cost'):
            if y is None:
                return None
            labels = tf.cast(y, tf.float32)

            tf.logging.info("logits: {};labels: {}".format(logits, labels))
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(per_example_loss)
            tf.logging.info("per_example_loss: {};reduce_mean_loss: {}".format(per_example_loss, loss))

            if l2_lambda > 0:
                weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]

                # L2 regularization
                l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
                loss += l2_loss

            return loss

    def multi_labe_onel_hot(self, prediction, threshold=0.5):
        prediction = tf.cast(prediction, tf.float32)
        threshold = float(threshold)
        return tf.cast(tf.greater(prediction, threshold), tf.int64)

    def _predict_op(self, y, logits):
        num_labels = self.model_config["num_labels"]
        sentence_fragment_length = self.model_config["sentence_fragment_length"]

        with tf.name_scope('acc'):
            probabilities = tf.nn.sigmoid(logits)
            print_shape('(_predict_op) probabilities', probabilities)
            label_pred = self.multi_labe_onel_hot(probabilities)
            print_shape('(_predict_op) label_pred', label_pred)

            batch_size = self.model_config["batch_size"]

            labels_flat = tf.reshape(y, [batch_size * sentence_fragment_length, num_labels])
            probabilities_flat = tf.reshape(probabilities, [batch_size * sentence_fragment_length, num_labels])
            labels_flat_pred = self.multi_labe_onel_hot(probabilities_flat)
            print_shape('(_predict_op) labels_flat_pred', labels_flat_pred)
            print_shape('(_predict_op) labels_flat', labels_flat)

            correct_pred = tf.equal(tf.cast(labels_flat_pred, tf.int32), tf.cast(labels_flat, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')

            return accuracy, labels_flat_pred, probabilities_flat, probabilities

    def seq_pair_classifier(self, input_ids, input_mask, segment_ids):
        print('seq_pair_classifier')
        bert_module = hub.Module(
            self.model_config["bert_model_hub"],
            trainable=True)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        seq_representation = bert_outputs["pooled_output"]
        print_shape('seq_representation', seq_representation)

        return seq_representation

    def cond(self, step, output):
        return step < self.model_config["sentence_fragment_length"]

    def body(self, step, output):
        max_sequence_length = self.model_config["max_sequence_length"]
        input_ids = tf.gather(ids_array, step)
        input_mask = tf.gather(mask_array, step)
        segment_ids = tf.gather(segment_array, step)
        # label_ids = tf.gather(label_array, step)

        input_ids = tf.reshape(input_ids, (-1, max_sequence_length))
        input_mask = tf.reshape(input_mask, (-1, max_sequence_length))
        segment_ids = tf.reshape(segment_ids, (-1, max_sequence_length))

        print_shape('body:\ninput_ids', input_ids)
        print_shape('input_mask', input_mask)
        print_shape('segment_ids', segment_ids)

        seq_pair_output = self.seq_pair_classifier(input_ids, input_mask, segment_ids)
        output = output.write(step, seq_pair_output)

        return step + 1, output

    def _positional_encoding_offset(self, inputs,
                                    maxlen,
                                    masking=True,
                                    scope="positional_encoding",
                                    offset_position=0):
        """
        Sinusoidal Positional_Encoding.
           PE is a two-dimensional matrix, the size is the same as the dimension
           of the input embedding, the row represents the word, and the column
           represents the word vector.
           pos indicates the position of the word in the sentence.
        Args:
           inputs: 3d tensor. (N, T, E)
           maxlen: scalar. Must be >= T
           masking: Boolean. If True, padding positions are set to zeros.
           scope: Optional scope for `variable_scope`.

        Returns:
            3d tensor that has the same shape as inputs.
        """

        E = inputs.get_shape().as_list()[-1]  # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [(pos + offset_position) / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.to_float(outputs)

    def _get_mini_batch_start_end(self, n_train, batch_size=None):
        """
        Args:
            n_train: int, number of training instances
            batch_size: int (or None if full batch)

        Returns:
            batches: list of tuples of (start, end) of each mini batch
        """
        mini_batch_size = n_train if batch_size is None else batch_size
        batches = zip(
            range(0, n_train, mini_batch_size),
            list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
        )
        return batches

    def create_feed_dict(self, is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        feed_dict = {
            self.input_ids: feed_input_ids,
            self.input_mask: feed_input_mask,
            self.segment_ids: feed_segment_ids
        }
        if is_train:
            feed_dict[self.label_ids] = feed_label_ids
            # feed_dict[self.dropout_keep_prob] = self.config["train_keep_prob"]
            # feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_init(self, sess):
        sess.run(self.init_global_op)
        sess.run(self.init_local_op)

    def run_step(self, sess, is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids=None):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """

        feed_dict = self.create_feed_dict(is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            logits, logits_flat, probabilities, probabilities_flat = sess.run(
                [self.logits, self.logits_flat, self.probabilities, self.probabilities_flat],
                feed_dict)
            return logits, logits_flat, probabilities, probabilities_flat

    def save_restore_model_init(self, model_save_dir):
        with tf.name_scope('save_model'):
            self.saver = tf.train.Saver(max_to_keep=20)
            self.model_save_dir = model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

    def save_model(self, sess, current_train_epoch):
        model_save_dir = self.model_save_dir + str(current_train_epoch) + '/'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            save_path = self.saver.save(sess, model_save_dir + 'model.ckpt')

    def restore_model(self, sess, model_file):
        self.saver.restore(sess, model_file)

    def evaluate(self,
                 sess,
                 example_all_input_ids,
                 example_all_input_mask,
                 example_all_segment_ids,
                 batch_size, sample_size):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """

        results = []
        cut_sample_size = sample_size - (sample_size % batch_size)

        for start_i in range(0, cut_sample_size, batch_size):
            end_i = start_i + batch_size
            feed_input_ids = example_all_input_ids[start_i:end_i]
            feed_input_mask = example_all_input_mask[start_i:end_i]
            feed_segment_ids = example_all_segment_ids[start_i:end_i]

            logits, logits_flat, probabilities, probabilities_flat = self.run_step(sess, False,
                                                                                   feed_input_ids,
                                                                                   feed_input_mask,
                                                                                   feed_segment_ids)

            results.extend(probabilities_flat)
        return results
