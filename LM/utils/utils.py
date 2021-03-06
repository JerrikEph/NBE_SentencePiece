
import _pickle as pkl
import pdb
import numpy as np
import copy

import os
import warnings
import sys
from time import time
import pprint
import logging
from collections import OrderedDict

'''check alive'''
def write_status(path, finished=False):
    full_path = path+'/status'
    if not finished:
        fd = open(full_path, 'w')
        fd.write(str(time()))
        fd.flush()
        fd.close()
    else:
        fd = open(full_path, 'w')
        fd.write('0.1')
        fd.flush()
        fd.close()

def read_status(status_path):
    if not os.path.exists(status_path):
        return 'error'
    fd = open(status_path, 'r')
    time_stamp = float(fd.read().strip())
    fd.close()
    if time_stamp < 10.:
        return 'finished'
    cur_time = time()
    if cur_time - time_stamp < 1000.:
        return 'running'
    else:
        return 'error'

def valid_entry(save_path):

    if not os.path.exists(save_path):
        return False
    if read_status(save_path + '/status') == 'running':
        return True
    if read_status(save_path + '/status') == 'finished':
        return True
    if read_status(save_path + '/status') == 'error':
        return False

    raise ValueError('unknown error')

'''Read and make embedding'''

def readEmbedding(fileName):
    """
    Read Embedding Function

    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'rb') as f:
        for line in f:
            line_uni = line.strip()
            line_uni = line_uni.decode('utf-8')
            values = line_uni.split(' ')
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                print(values, len(values))
            embeddings_index[word] = coefs
    return embeddings_index

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix

    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) < 1:
        raise ValueError('Input dimension less than 1')
    vocab_sz = max(vocab_dic.values()) + 1
    EMBEDDING_DIM = len(list(embed_dic.values())[0])
    # embedding_matrix = np.zeros((len(vocab_dic), EMBEDDING_DIM), dtype=np.float32)
    embedding_matrix = np.random.rand(vocab_sz, EMBEDDING_DIM).astype(np.float32) * 0.05
    valid_mask = np.ones(vocab_sz, dtype=np.bool)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            valid_mask[i] = False
    return embedding_matrix, valid_mask

'''evaluation'''

def pred_from_prob_single(prob_matrix):
    """

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num),
            type of float. Generated from softmax activation

    Returns:
        ret: return class ids, shape of(data_num,)
    """
    ret = np.argmax(prob_matrix, axis=1)
    return ret


def calculate_accuracy_single(pred_ids, label_ids):
    """
    Args:
        pred_ids: prediction id list shape of (data_num, ), type of int
        label_ids: true label id list, same shape and type as pred_ids

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if np.ndim(pred_ids) != 1 or np.ndim(label_ids) != 1:
        raise TypeError('require rank 1, 1. get {}, {}'.format(np.rank(pred_ids), np.rank(label_ids)))
    if len(pred_ids) != len(label_ids):
        raise TypeError('first argument and second argument have different length')

    accuracy = np.mean(np.equal(pred_ids, label_ids))
    return accuracy


def calculate_confusion_single(pred_list, label_list, label_size):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((label_size, label_size), dtype=np.int32)
    for i in range(len(label_list)):
        confusion[label_list[i], pred_list[i]] += 1

    tp_fp = np.sum(confusion, axis=0)
    tp_fn = np.sum(confusion, axis=1)
    tp = np.array([confusion[i, i] for i in range(len(confusion))])

    precision = tp.astype(np.float32)/(tp_fp+1e-40)
    recall = tp.astype(np.float32)/(tp_fn+1e-40)
    overall_prec = np.float(np.sum(tp))/(np.sum(tp_fp)+1e-40)
    overall_recall = np.float(np.sum(tp))/(np.sum(tp_fn)+1e-40)

    return precision, recall, overall_prec, overall_recall, confusion


def print_confusion_single(prec, recall, overall_prec, overall_recall, num_to_tag):
    """Helper method that prints confusion matrix."""
    logstr="\n"
    logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
    for i, tag in sorted(num_to_tag.items()):
        logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag, prec[i], recall[i])
    logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format('OVERALL', overall_prec, overall_recall)
    logging.info(logstr)
    print(logstr)


def save_objs(obj, path):
    with open(path, 'wb') as fd:
        pkl.dump(obj, fd)


def load_objs(path):
    with open(path, 'rb') as fd:
        ret = pkl.load(fd)
    return ret