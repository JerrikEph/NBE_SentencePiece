'''
Created on Sep 21, 2016

@author: jerrik
'''

import os, sys, time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from .utils import utils, nest
from .utils.TfUtils import entry_stop_gradients, mkMask, reduce_avg, masked_softmax

NINF = -1e20
EPSILON = 1e-10

class model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, config):
        """options in this function"""
        self.config = config
        self.EX_REG_SCOPE = []

        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_accu = tf.assign_add(self.on_epoch, 1)


    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.
        inputs: a list of tensors each of which have a size of [batch_size, embed_size]
        """
        self.global_step = tf.train.get_or_create_global_step()
        vocab_sz = self.config.vocab_sz
        with tf.variable_scope('embedding') as scp:
            self.exclude_reg_scope(scp)
            if self.config.pre_trained:
                embed = utils.readEmbedding(self.config.embed_path)
                embed_matrix, valid_mask = utils.mkEmbedMatrix(embed, dict(self.config.vocab_dict))
                embedding = tf.Variable(embed_matrix, 'Embedding')
                partial_update_embedding = entry_stop_gradients(embedding, tf.expand_dims(valid_mask, 1))
                embedding = tf.cond(self.on_epoch < self.config.partial_update_until_epoch,
                                    lambda: partial_update_embedding, lambda: embedding)
            else:
                embedding = tf.get_variable(
                  'Embedding',
                  [vocab_sz, self.config.embed_size], trainable=True)
        return embedding

    def embed_lookup(self, embedding, batch_x, dropout=None, is_train=False):
        '''

        :param embedding: shape(v_sz, emb_sz)
        :param batch_x: shape(b_sz, wNum)
        :return: shape(b_sz, wNum, emb_sz)
        '''
        inputs = tf.nn.embedding_lookup(embedding, batch_x)
        if dropout is not None:
            inputs = tf.layers.dropout(inputs, rate=dropout, training=is_train)
        return inputs

    def add_loss_op(self, logits, labels, xlen):
        '''

        :param logits: [shape(b_sz, xlen, c_num), ..] type(float)
        :param labels: [shape(b_sz,xlen), ..] type(int)
        :param xlen: shape(b_sz)
        :return:
        '''

        loss_fwd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[0],
                                                                  labels=labels[0])  # shape(b_sz, xlen)
        loss_fwd = reduce_avg(loss_fwd, xlen, dim=1)  # shape(b_sz,)

        loss_bwd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[1],
                                                                  labels=labels[1])  # shape(b_sz, xlen)
        loss_bwd = reduce_avg(loss_bwd, xlen, dim=1)  # shape(b_sz,)

        ce_loss = tf.reduce_mean(loss_fwd) + tf.reduce_mean(loss_bwd)
        # ce_loss = tf.reduce_mean(loss)

        exclude_vars = nest.flatten([[v for v in tf.trainable_variables(o.name)] for o in self.EX_REG_SCOPE])
        exclude_vars_2 = [v for v in tf.trainable_variables() if '/bias:' in v.name]
        exclude_vars = exclude_vars + exclude_vars_2

        reg_var_list = [v for v in tf.trainable_variables() if v not in exclude_vars]
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_var_list])
        self.param_cnt = np.sum([np.prod(v.get_shape().as_list()) for v in reg_var_list])

        print('===' * 20)
        print('total reg parameter count: %.3f M' % (self.param_cnt / 1000000.))
        print('excluded variables from regularization')
        print([v.name for v in exclude_vars])
        print('===' * 20)

        print('regularized variables')
        print(['%s:%.3fM' % (v.name, np.prod(v.get_shape().as_list()) / 1000000.) for v in reg_var_list])
        print('===' * 20)
        '''shape(b_sz,)'''
        self.ce_loss = ce_loss
        reg = self.config.reg

        return self.ce_loss + reg * reg_loss
        # return tf.reduce_mean(self.ce_loss)

    def add_train_op(self, loss):

        lr = tf.train.exponential_decay(self.config.lr, self.global_step,
                                        self.config.decay_steps,
                                        self.config.decay_rate, staircase=True)
        self.learning_rate = tf.maximum(lr, 1e-5)
        if self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config.optimizer == 'grad':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adgrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        else:
            raise ValueError('No such Optimizer: %s' % self.config.optimizer)

        gvs = optimizer.compute_gradients(loss=loss)

        capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        return train_op

    def exclude_reg_scope(self, scope):
        if scope not in self.EX_REG_SCOPE:
            self.EX_REG_SCOPE.append(scope)

    def __call__(self, features, labels, mode, params):

        is_train = True if mode == tf.estimator.ModeKeys.TRAIN else False
        ph_input = features['f_wids']   # shape(b_sz, xlen)
        ph_len = features['f_len']      # shape(b_sz)
        fwd_label = labels['l_fwd']
        bwd_label = labels['l_bwd']
        self.embedding = self.add_embedding()
        vocab_sz = self.embedding.get_shape()[0]
        '''shape(b_sz, xlen, emb_sz)'''
        in_x = self.embed_lookup(self.embedding, ph_input,
                                 dropout=self.config.dropout, is_train=is_train)
        '''[shape(b_sz, xlen, h_sz), ...]'''
        out_x = self.biLSTM(in_x=in_x, xLen=ph_len, h_sz=self.config.hidden_size, scope='biLSTM')
        out_x = (tf.layers.dense(out_x[0], self.config.embed_size, activation=tf.nn.tanh, name='dense-fwd'),
                 tf.layers.dense(out_x[1], self.config.embed_size, activation=tf.nn.tanh, name='dense-bwd'))
        '''[shape(b_sz, xlen, v_sz), ...]'''
        fwd_bias = tf.get_variable('fwddigit_bias', shape=[vocab_sz], dtype=self.embedding.dtype,
                                   initializer=tf.zeros_initializer)
        bwd_bias = tf.get_variable('bwddigit_bias', shape=[vocab_sz], dtype=self.embedding.dtype,
                                   initializer=tf.zeros_initializer)
        out_digits = (tf.tensordot(out_x[0], self.embedding, axes=[[-1], [-1]])+fwd_bias,
                      tf.tensordot(out_x[1], self.embedding, axes=[[-1], [-1]])+bwd_bias)

        opt_loss = self.add_loss_op(out_digits, (fwd_label, bwd_label),xlen=ph_len)
        train_op = self.add_train_op(opt_loss)
        self.train_op = train_op
        self.opt_loss = opt_loss
        tf.summary.scalar('ce_loss', self.ce_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode, loss=opt_loss, train_op=train_op)

    @staticmethod
    def biLSTM(in_x, xLen, h_sz, scope=None):

        with tf.variable_scope(scope or 'biLSTM'):
            cell_fwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen,
                                                       dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')
        return x_out

    @staticmethod
    def biGRU(in_x, xLen, h_sz, scope=None):

        with tf.variable_scope(scope or 'biGRU'):
            cell_fwd = tf.nn.rnn_cell.GRUCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.GRUCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen,
                                                       dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')
        return x_out
