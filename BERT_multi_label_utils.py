# -*- coding: utf-8 -*-
# @Time : 2021/8/26 19:38
# @Author : luff543
# @Email : luff543@gmail.com
# @File : BERT_multi_label_utils.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow_hub as hub
import bert

# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))


def create_tokenizer_from_hub_module(bert_model_hub):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
