# -*- coding: utf-8 -*-
# @Time : 2022/9/26 19:21
# @Author : luff543
# @Email : luff543@gmail.com
# @File : train.py
# @Software: PyCharm

import argparse
from engines.utils.io import fold_check
from engines.utils.logger import get_logger
from engines.configure import Configure
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from CABERT_model_finetune import CABERTModel
import sklearn.metrics as skm
from bert import run_multi_label_classifier
from bert import tokenization
import os
import BERT_multi_label_utils as BERT_multi_label_utils
from tensorflow.contrib.layers.python.layers import initializers

SENTENCE1_COLUMN = 's1'
SENTENCE2_COLUMN = 's2'
LABEL_COLUMN = 'label'
LABELS_DICT = {}
NUM_LABELS = 0


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyProcessor(DataProcessor):

    def get_set_examples(self, data_path):
        return self.create_examples(
            self._read_tsv(data_path), "set")

    def get_test_examples(self, data_dir):
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_pred_examples(self, data_dir):
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred")

    def get_labels(self):
        """See base class."""
        return ["T", "V", "SD", "ED"]

    def create_examples(self, lines, set_type, file_base=True):
        """Creates examples for the training and dev sets. each line is label+\t+text_a+\t+text_b """
        examples = []
        for (i, line) in tqdm(enumerate(lines)):

            if file_base:
                if i == 0:
                    continue

            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            if set_type == "test" or set_type == "pred":
                label = tokenization.convert_to_unicode(line[0])

                labels = label.split(",")
                one_hot_labels = np.zeros(NUM_LABELS).astype(int)
                for current_label in labels:
                    one_hot_labels[LABELS_DICT[current_label]] = 1

            else:
                label = tokenization.convert_to_unicode(line[0])

                labels = label.split(",")
                one_hot_labels = np.zeros(NUM_LABELS).astype(int)
                for current_label in labels:
                    one_hot_labels[LABELS_DICT[current_label]] = 1

            examples.append(
                InputExample(guid=guid, text_a=text, label=one_hot_labels))
        return examples


def load_dataset(dataset_path, sentence_fragment_length=10):
    df = pd.DataFrame()
    s1_list, s2_list, label_list = [], [], []
    max_doc_len = 10
    line_num = 0
    with open(dataset_path, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('-DOCSTART-'):
                continue
            try:
                l, s1, s2 = [v.strip() for v in line.strip().split('|||')]

                s1 = [v.strip() for v in s1.split()]
                s2 = [v.strip() for v in s2.split()]

                if len(s1) != 1:
                    s1 = "﹍"
                    s1 = [v.strip() for v in s1.split()]
                if len(s2) != 1:
                    s2 = "﹍"
                    s2 = [v.strip() for v in s2.split()]

                labels = l.split(",")
                one_hot_labels = np.zeros(NUM_LABELS).astype(int)
                for current_label in labels:
                    if current_label in LABELS_DICT:
                        one_hot_labels[LABELS_DICT[current_label]] = 1

                if (len(labels) > 1):
                    one_hot_labels = np.zeros(NUM_LABELS).astype(int)
                    for current_label in labels:
                        one_hot_labels[LABELS_DICT[current_label]] = 1
                # 0, 1, 2, 3
                if line_num % max_doc_len < sentence_fragment_length:
                    s1_list.append(s1[0])
                    s2_list.append(s2[0])
                    label_list.append(one_hot_labels)
                line_num = line_num + 1
            except Exception:
                logger.info("line_num error: " + str(line_num))
                ValueError('Input Data Value Error!')

    df[SENTENCE1_COLUMN] = s1_list
    df[SENTENCE2_COLUMN] = s2_list
    df[LABEL_COLUMN] = label_list
    return df


def get_one_hot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def restore_training_task(save_model_dir):
    model_paths = []
    for dir_path, _, filenames in os.walk(save_model_dir):
        if filenames:
            for filename in filenames:
                if filename.endswith(".meta"):
                    model_name = os.path.splitext(filename)[0]
                    model_paths.append(os.path.abspath(os.path.join(dir_path, model_name)))
    model_paths = sorted(model_paths, key=lambda x: int(x.split("/")[-2]))
    restore_model_path = model_paths[-1]
    restore_epoch = int(model_paths[-1].split("/")[-2])

    return restore_epoch, restore_model_path


def restore_training_model(sess, save_model_dir):
    model_paths = []
    for dir_path, _, filenames in os.walk(save_model_dir):
        if filenames:
            for filename in filenames:
                if filename.endswith(".meta"):
                    model_name = os.path.splitext(filename)[0]
                    model_paths.append(os.path.abspath(os.path.join(dir_path, model_name)))
    model_paths = sorted(model_paths, key=lambda x: int(x.split("/")[-2]))
    restore_model_path = model_paths[-1]
    restore_epoch = int(model_paths[-1].split("/")[-2])
    imported_meta = tf.train.import_meta_graph(
        os.path.join(restore_model_path + '.meta'))
    imported_meta.restore(sess, restore_model_path)
    return restore_epoch


def config_model(configs, logger):
    model_config = OrderedDict()

    model_config["batch_size"] = configs.batch_size
    max_sequence_length = configs.max_sequence_length
    all_labels = configs.all_labels
    num_labels = len(configs.all_labels)

    model_config["all_labels"] = all_labels
    model_config["num_labels"] = num_labels
    model_config["max_sequence_length"] = max_sequence_length
    model_config["sentence_fragment_length"] = configs.sentence_fragment_length
    model_config["bert_model_hub"] = configs.bert_model_hub
    model_config["num_gru_layers"] = configs.num_gru_layers
    model_config["gru_hidden_size"] = configs.gru_hidden_size
    model_config["dropout_keep_prob"] = configs.dropout_keep_prob
    model_config["l2_lambda"] = configs.l2_lambda


    model_config["input_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, None, max_sequence_length],
                                         name="input_ids")
    model_config["input_mask"] = tf.placeholder(dtype=tf.int32, shape=[None, None, max_sequence_length],
                                          name="input_mask")
    model_config["segment_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, None, max_sequence_length],
                                           name="segment_ids")
    model_config["label_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, None, num_labels],
                                         name="label_ids")
    model_config["global_step"] = tf.Variable(0, trainable=False)
    model_config["initializers"] = initializers
    model_config["logger"] = logger

    return model_config


def convert_data_to_examples(data_input_examples, labels, tokenizer, model_config):
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    label_ids_list = []

    sentence_fragment_length = model_config["sentence_fragment_length"]
    max_sequence_length = model_config["max_sequence_length"]

    for (ex_index, example) in enumerate(data_input_examples):
        feature = run_multi_label_classifier.convert_single_example(ex_index, example, labels,
                                                                    max_sequence_length, tokenizer)
        input_ids_list.append(feature.input_ids)
        input_mask_list.append(feature.input_mask)
        segment_ids_list.append(feature.segment_ids)
        label_ids_list.append(feature.label_id)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

    num_examples = int(len(data_input_examples) / sentence_fragment_length)
    all_input_ids = np.reshape(input_ids_list, (num_examples, sentence_fragment_length, max_sequence_length))
    all_input_mask = np.reshape(input_mask_list, (num_examples, sentence_fragment_length, max_sequence_length))
    all_segment_ids = np.reshape(segment_ids_list,
                                 (num_examples, sentence_fragment_length, max_sequence_length))
    all_label_ids = np.reshape(label_ids_list, (num_examples, sentence_fragment_length, NUM_LABELS))
    example_size = int(len(data_input_examples) / sentence_fragment_length)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, example_size


def read_dataset_scores(path):
    scores = []
    fp = open(path, "r")
    line = fp.readline()
    while line:
        score = []
        # logger.info(line)
        split_line = line.split("\t")
        for current_prob in split_line:
            score.append(float(current_prob))

        line = fp.readline()
        scores.append(score)

    return scores


def evaluation(dataset_path, predict_path, model_config, save_result_path=None):
    sentence_fragment_length = model_config["sentence_fragment_length"]
    threshold = 0.5

    scores = read_dataset_scores(predict_path)

    dataset = load_dataset(dataset_path, sentence_fragment_length)

    input_examples = dataset.apply(lambda x: run_multi_label_classifier.InputExample(guid=None,
                                                                                     text_a=x[
                                                                                         SENTENCE1_COLUMN],
                                                                                     label=x[LABEL_COLUMN]),
                                   axis=1)

    gold_labels = []
    for example in input_examples:
        gold_labels.append(example.label)

    scores = np.array(scores)

    y_test = np.asarray(gold_labels)
    y_pred = get_one_hot_label_threshold(scores, threshold)

    if (len(y_pred) != len(y_test)):
        y_pred_padding = np.zeros((len(y_test) - len(y_pred)) * NUM_LABELS)
        y_pred_padding = np.reshape(y_pred_padding, [len(y_test) - len(y_pred), NUM_LABELS])
        y_pred = np.concatenate((y_pred, y_pred_padding), axis=0)

    cm = skm.multilabel_confusion_matrix(y_test, y_pred)
    result = skm.classification_report(y_test, y_pred, digits=4)

    macro_avg_f1 = 0
    if save_result_path is not None:
        fp = open(save_result_path, "a")
        fp.write("confusion matrix: " + str(cm) + "\n")
        fp.write("performance: " + str(result) + "\n")

        result_split_line = result.split("\n")
        macro_avg_line = result_split_line[8]
        macro_avg_line = macro_avg_line.replace("    ", " ").replace("   macro avg  ", "macroavg ")
        macro_avg_line_split = macro_avg_line.split(" ")
        if macro_avg_line_split[0] == "macroavg":
            macro_avg_f1 = float(macro_avg_line_split[3])
        fp.close()

    return macro_avg_f1


def evaluation_model(dataset_path, save_dir, epoch, dataset_name, model_config):
    batch_size = model_config["batch_size"]
    sentence_fragment_length = model_config["sentence_fragment_length"]
    all_labels = model_config["all_labels"]

    predict_path = os.path.join(save_dir, dataset_name + "_results_" + str(epoch) + ".tsv")
    dataset = load_dataset(dataset_path, sentence_fragment_length)

    input_examples = dataset.apply(lambda x: run_multi_label_classifier.InputExample(guid=None,
                                                                                     text_a=x[SENTENCE1_COLUMN],
                                                                                     label=x[LABEL_COLUMN]),
                                   axis=1)
    example_input_ids, example_all_input_mask, example_all_segment_ids, example_all_label_ids, sample_size = convert_data_to_examples(
        input_examples, all_labels, tokenizer, model_config)

    predict_results = ca_BERT_model.evaluate(sess, example_input_ids, example_all_input_mask,
                                             example_all_segment_ids, batch_size, sample_size)

    output_predict_file = os.path.join(predict_path)

    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for prediction in predict_results:
            output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)

    return predict_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CA-BERT-MLP')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)
    fold_check(configs)
    # logger = get_logger(configs.log_dir)
    logger = get_logger(log_dir=configs.log_dir)
    configs.show_data_summary(logger)
    model_config = config_model(configs, logger)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs.CUDA_VISIBLE_DEVICES)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    gpu_options.allow_growth = True
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=tf_config)

    workspace = configs.workspace

    all_labels = configs.all_labels
    NUM_LABELS = len(all_labels)
    batch_size = configs.batch_size
    num_train_epochs = configs.epoch
    max_sequence_length = configs.max_sequence_length
    sentence_fragment_length = configs.sentence_fragment_length
    warmup_proportion = configs.bert_warmup_proportion


    for (i, label) in enumerate(all_labels):
        LABELS_DICT[label] = i

    tokenizer = BERT_multi_label_utils.create_tokenizer_from_hub_module(configs.bert_model_hub)

    data_dir = workspace + configs.datasets_fold
    output_dir = workspace + configs.output_dir

    # Load train-dev data    
    train_dataset_path = os.path.join(data_dir, "train.txt")
    dev_dataset_path = os.path.join(data_dir, "dev.txt")
    test_dataset_path = os.path.join(data_dir, "test.txt")

    is_save_model = True
    is_restore_model = False

    model_save_dir = workspace + configs.model_fold + "/"

    train = load_dataset(train_dataset_path, sentence_fragment_length)

    train_input_examples = train.apply(lambda x: run_multi_label_classifier.InputExample(guid=None,
                                                                                         text_a=x[
                                                                                             SENTENCE1_COLUMN],
                                                                                         label=x[LABEL_COLUMN]),
                                       axis=1)

    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_label_ids = []

    for (ex_index, example) in enumerate(train_input_examples):
        feature = run_multi_label_classifier.convert_single_example(ex_index, example, all_labels,
                                                                    max_sequence_length, tokenizer)
        train_input_ids.append(feature.input_ids)
        train_input_mask.append(feature.input_mask)
        train_segment_ids.append(feature.segment_ids)
        train_label_ids.append(feature.label_id)


        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

    num_examples = int(len(train_input_examples) / sentence_fragment_length)
    logger.info('*' * 20 + "train dataset" + '*' * 20)
    logger.info("features: {}".format(len(train_input_examples)))
    logger.info("num_examples: {}".format(num_examples))
    train_example_input_ids = np.reshape(train_input_ids, (num_examples, sentence_fragment_length, max_sequence_length))
    train_example_input_mask = np.reshape(train_input_mask,
                                          (num_examples, sentence_fragment_length, max_sequence_length))
    train_example_segment_ids = np.reshape(train_segment_ids,
                                           (num_examples, sentence_fragment_length, max_sequence_length))
    train_example_label_ids = np.reshape(train_label_ids, (num_examples, sentence_fragment_length, NUM_LABELS))
    train_example_size = int(len(train_input_examples) / sentence_fragment_length)

    num_train_steps = int(int(len(train_input_examples) / sentence_fragment_length) / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    dev = load_dataset(dev_dataset_path, sentence_fragment_length)

    dev_input_examples = dev.apply(lambda x: run_multi_label_classifier.InputExample(guid=None,
                                                                                     text_a=x[SENTENCE1_COLUMN],
                                                                                     label=x[LABEL_COLUMN]),
                                   axis=1)
    dev_example_input_ids, dev_example_all_input_mask, dev_example_all_segment_ids, dev_example_all_label_ids, dev_sample_size = convert_data_to_examples(
        dev_input_examples, all_labels, tokenizer, model_config)


    model_config["num_train_steps"] = num_train_steps
    model_config["num_warmup_steps"] = num_warmup_steps
    model_config["is_training"] = True

    restore_epoch = 0
    restore_model_path = ""
    if is_restore_model:
        restore_epoch, restore_model_path = restore_training_task(model_save_dir)

    current_best_macroavgf1 = 0.0
    with tf.compat.v1.Session(
            config=tf_config) as sess:
        ca_BERT_model = CABERTModel(model_config)
        if not is_restore_model:
            ca_BERT_model.run_init(sess)
        if is_save_model or is_restore_model:
            ca_BERT_model.save_restore_model_init(model_save_dir)
        if is_restore_model:
            restore_epoch = restore_training_model(sess, model_save_dir)

        for t in range(restore_epoch + 1, num_train_epochs):
            total_loss = 0.0
            batch_loss_list = []
            cut_train_sample_size = train_example_size - (train_example_size % batch_size)
            logger.info("cut_train_sample_size: : {}".format(cut_train_sample_size))
            for start_i in range(0, cut_train_sample_size, batch_size):
                end_i = start_i + batch_size
                feed_input_ids = train_example_input_ids[start_i:end_i]
                feed_input_mask = train_example_input_mask[start_i:end_i]
                feed_segment_ids = train_example_segment_ids[start_i:end_i]
                feed_label_ids = train_example_label_ids[start_i:end_i]

                global_step, batch_loss = ca_BERT_model.run_step(sess, True,
                                                                 feed_input_ids,
                                                                 feed_input_mask,
                                                                 feed_segment_ids,
                                                                 feed_label_ids)
                total_loss += batch_loss
                batch_loss_list.append(batch_loss)
            logger.info("epoch: {}".format(t))
            logger.info("total loss: : {}, mean loss: {}".format(total_loss, np.mean(batch_loss_list)))

            predict_val_peformance_path = os.path.join(output_dir, "validation_performance_" + str(t) + ".txt")
            predict_val_dataset_path = evaluation_model(dev_dataset_path, output_dir, t, "validation", model_config)
            macroavgf1 = evaluation(dev_dataset_path, predict_val_dataset_path,
                                    model_config, predict_val_peformance_path)

            predict_test_peformance_path = os.path.join(output_dir, "test_performance_" + str(t) + ".txt")
            predict_test_dataset_path = evaluation_model(test_dataset_path, output_dir, t, "test", model_config)
            evaluation(test_dataset_path, predict_test_dataset_path, model_config, predict_test_peformance_path)

            if is_save_model and current_best_macroavgf1 < macroavgf1:
                current_best_macroavgf1 = macroavgf1
                ca_BERT_model.save_model(sess, t)

                fp = open(os.path.join(output_dir, "best_performance_model.txt"), "a")
                info = "epoch: {}\nmacro avg f1: {}\n".format(t, macroavgf1)
                fp.write(info)
                fp.close()
                logger.info(info)
