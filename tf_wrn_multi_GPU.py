"""
Trainable CHiME-4 Wide Residual BLSTM Network (WRBN) with Multi GPU Support

1. This model does not have dropouts for hidden-hidden transitions of LSTM.
2. The author used lists to do batch training. Better implementations are welcomed.
3. The author recommends to use queues to read data.

Author: Peidong Wang
Date:   20170723
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


### Normalization ###
def normalize_3d(x, gamma, beta, utt_len):
    with tf.name_scope('normalize', values=[x, gamma, beta]):
        mean = [None for i in range(utt_len.shape[0])]
        var = [None for i in range(utt_len.shape[0])]
        
        # calculate moments for each utterance (trimmed)
        for utt in range(utt_len.shape[0]):
            mean[utt], var[utt] = tf.nn.moments(x[:utt_len[utt], utt, :][:,None,:], axes=(0,), keep_dims=True)
        
        mean_tot = tf.concat(mean, axis=1)
        var_tot = tf.concat(var, axis=1)
        
        # batch normalize and apply sequence mask on the results
        return tf.multiply(tf.transpose(tf.sequence_mask(utt_len, dtype=tf.float32))[:, :, None], tf.nn.batch_normalization(x, mean_tot, var_tot, beta[None, None, :], gamma[None, None, :], 1e-8))


def normalize_4d(x, scope, utt_len, initializer):
    with tf.variable_scope(scope, values=[x]):
        channels = x.get_shape().as_list()[-1]
        gamma = tf.get_variable('gamma', initializer=tf.convert_to_tensor(initializer[scope+'/gamma']))
        beta = tf.get_variable('beta', initializer=tf.convert_to_tensor(initializer[scope+'/beta']))

    with tf.name_scope(scope.split('/')[-1], values=[x, gamma, beta]):
        mean = [None for i in range(utt_len.shape[0])]
        var = [None for i in range(utt_len.shape[0])]

        for utt in range(utt_len.shape[0]):
            mean[utt], var[utt] = tf.nn.moments(x[utt, :, :utt_len[utt], :][None,...], axes=(1,2), keep_dims=True)

        mean_tot = tf.concat(mean, axis=0)
        var_tot = tf.concat(var, axis=0)

        return tf.multiply(tf.sequence_mask(utt_len, dtype=tf.float32)[:,None,:,None], tf.nn.batch_normalization(x, mean_tot, var_tot, beta, gamma, 1e-5))


### Conv ###
def conv_3_3(x, out_channels, strides, pads, scope, initializer):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        if np.sum(pads):
            x = tf.pad(x, ((0, 0), *pads, (0, 0)))
            cut = lambda x: x[:, :-1, :, :]
        else:
            cut = lambda x: x

        with tf.variable_scope(scope, values=[x]):
            _shape = [3, 3, x.get_shape().as_list()[3], out_channels]
            filters = tf.get_variable(name='W', initializer=tf.convert_to_tensor(initializer[scope + '/W'][:].transpose(2, 3, 1, 0)))

        z = tf.nn.conv2d(x, filters, strides=(1, *strides, 1), padding='SAME', data_format='NHWC')

        z = cut(z)

    return z


def conv_1_1(x, out_channels, strides, scope, initializer):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        ch_idx = 3
        strides = (1, *strides, 1)

        with tf.variable_scope(scope, values=[x]):
            _shape = [1, 1, x.get_shape().as_list()[ch_idx], out_channels]
            filters = tf.get_variable(name='W', initializer=tf.convert_to_tensor(initializer[scope + '/W'][:].transpose(2, 3, 1, 0)))

        z = tf.nn.conv2d(x, filters, strides=strides, padding='SAME',data_format='NHWC')

    return z


def wide_layer(x, out_channels, strides, scope, keep_prob, utt_len, initializer):
    with tf.name_scope(scope, values=[x]):
        in_channels = x.get_shape().as_list()[-1]
        x_norm = tf.nn.elu(normalize_4d(x, scope + '/bn1', utt_len, initializer))

        if strides[0] > 1:
            pads = ((1, 1), (0, 0))
        else:
            pads = ((0, 0), (0, 0))

        if in_channels != out_channels:
            ident = conv_1_1(x_norm, out_channels, strides, scope + '/conv_ident', initializer)
        else:
            ident = x

        h1 = conv_3_3(x_norm, out_channels, strides, pads, scope + '/conv1', initializer)
        h1 = tf.nn.elu(normalize_4d(h1, scope + '/bn2', utt_len, initializer))
        
        # dropout in residual block
        h1 = tf.nn.dropout(h1, keep_prob)
        h2 = conv_3_3(h1, out_channels, (1, 1), ((0, 0), (0, 0)), scope + '/conv2', initializer)

        return h2 + ident


def wide_block(x, layer, output_channels, strides, scope, keep_prob, utt_len, initializer):
    h = wide_layer(x, output_channels, strides, scope + '/a', keep_prob, utt_len, initializer)

    for l_idx in range(1, layer):
        h = wide_layer(h, output_channels, (1, 1), scope + '/b' + str(l_idx), keep_prob, utt_len, initializer)

    return h


### LSTM ###
def _lstm_step(x, c, h, num_units):
    a, i, f, o = tf.split(x, 4, axis=1)
    new_input = tf.nn.tanh(a)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)

    new_c = (c * f + i * new_input)
    new_h = tf.nn.tanh(new_c) * o

    return new_c, new_h


class LSTMCell(rnn.RNNCell):

    def __init__(self, num_units, W_h):
        self._num_units = num_units
        self._W_h = W_h

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        c, h = state

        inputs += tf.matmul(h, self._W_h)

        new_c, new_h = _lstm_step(inputs, c, h, self._num_units)
        new_state = (new_c, new_h)
        return new_h, new_state


def _stride(w):
    w_a = w[..., ::4]
    w_i = w[..., 1::4]
    w_f = w[..., 2::4]
    w_o = w[..., 3::4]
    return np.concatenate([w_a, w_i, w_f, w_o], axis=-1)


# note sequence_length here is the same as utt_len in inference()
def lstm(x, sequence_length, units, scope, reverse, keep_prob, initializer):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        # dropout on the input of LSTM
        x = tf.nn.dropout(x, keep_prob)
        
        if reverse:
            x = tf.reverse_sequence(x, sequence_length, seq_axis=0, batch_axis=1)
        
        in_size = x.get_shape().as_list()[-1]

        with tf.variable_scope(scope):
            gamma = tf.get_variable('gamma',initializer=tf.convert_to_tensor(_stride(initializer[scope + '/gamma'][:])))
            beta = tf.get_variable('beta', initializer=tf.convert_to_tensor(_stride(initializer[scope + '/beta'][:])))
            W_x = tf.get_variable('W_x', initializer=tf.convert_to_tensor(_stride(initializer[scope + '/W_x'][:])))
            W_h = tf.get_variable('W_h', initializer=tf.convert_to_tensor(_stride(initializer[scope + '/W_h'][:])))

        with tf.name_scope('input_transform'):
            x = normalize_3d(tf.einsum('tbf,fu->tbu', x, W_x), gamma, beta, sequence_length)
        
        cell = rnn.CompiledWrapper(LSTMCell(units, W_h))
        h, _ = tf.nn.dynamic_rnn(cell, x, time_major=True, dtype=x.dtype,sequence_length=sequence_length)
        
        if reverse:
            h = tf.reverse_sequence(h, sequence_length, seq_axis=0, batch_axis=1)
    return h


def blstm(x, sequence_length, stack_output, units, scope, keep_prob, initializer):
    with tf.name_scope(scope):
        fw_scope = scope + '/lstm_fw'
        bw_scope = scope + '/lstm_bw'
        h_fw = lstm(x, sequence_length, units, fw_scope, False, keep_prob, initializer)
        h_bw = lstm(x, sequence_length, units, bw_scope, True, keep_prob, initializer)

        if stack_output:
            h = tf.concat([h_fw, h_bw], axis=2)
        else:
            h = h_fw + h_bw

        return h


def stacked_blstm(x, sequence_length, units, scope, keep_prob, initializer):
    h = x
    
    for idx, num_units in enumerate(units):
        _scope = scope + '/' + str(idx)
        h = blstm(h, sequence_length, idx == len(units)-1, num_units,_scope, keep_prob, initializer)
    
    return h


### Dense ###
# dense_4d() is a slimmer implementation of the first "ELU + Linear" layer
def dense_4d(x, scope, activation, initializer):
    with tf.name_scope(scope.split('/')[-1], values=[x]):

        in_size = x.get_shape().as_list()[-1]

        with tf.variable_scope(scope):
            b_val = initializer[scope + '/b'][:]
            b = tf.get_variable('b', initializer=tf.convert_to_tensor(b_val))
            W = tf.get_variable('W', initializer=tf.transpose(tf.convert_to_tensor(initializer[scope + '/W'][:])))

        with tf.name_scope('input_transform'):
            x = tf.einsum('tbcf,fd->tbcd', x, W)
        x = b[None, None, :] + x[...,0]

        return activation(x)


def dense(x, units, scope, normalize, activation, utt_len, initializer, transformation='tbf,fu->tbu'):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        in_size = x.get_shape().as_list()[-1]
        
        with tf.variable_scope(scope):
            if normalize:
                gamma = tf.get_variable('gamma', initializer=tf.convert_to_tensor(initializer[scope + '/gamma'][:]))
                beta = tf.get_variable('beta', initializer=tf.convert_to_tensor(initializer[scope + '/beta'][:]))
            else:
                b_val = initializer[scope + '/b'][:]
                b = tf.get_variable('b', initializer=tf.convert_to_tensor(b_val))
            W = tf.get_variable('W', initializer=tf.convert_to_tensor(initializer[scope + '/W'][:]))

        with tf.name_scope('input_transform'):
            x = tf.einsum(transformation, x, W)
        
        if normalize:
            x = normalize_3d(x, gamma, beta, utt_len)
        else:
            x += b[None, None, :]

    return activation(x)


def stacked_dense(x, units, scope, utt_len, initializer):
    with tf.name_scope(scope):
        h = x
        
        for l_idx, num_units in enumerate(units):
            if  l_idx == len(units) - 1:
                activation = lambda x: x
            else:
                activation = tf.nn.elu
            h = dense(h, num_units, scope + '/' + str(l_idx), l_idx != len(units) - 1, activation, utt_len, initializer)

    return h


### Inference and Loss Calculation ###
def inference(x, utt_len, batch_size, num_gpus, keep_prob, initializer):
    """
    x: data (B x T x 240), where T is the maximal utterance length in the batch
    utt_len: one dimensional array (B) of the utterance lengths in the batch
    batch_size: the total batch size for all the GPUs
    num_gpus: the number of GPUs you used
    keep_prob: the keep probability for dropouts
    initializer: a dictionary of the model variables, the same to the one in build_resnet() of tf_wrn.py
    """

    with tf.name_scope('to_bhwc'):
        _feature_dim = 240 # x.get_shape().as_list()[2]
        x = tf.reshape(x, (batch_size//num_gpus, -1, 3, _feature_dim // 3))
        x = tf.transpose(x, (0, 3, 1, 2))

    # Input Convolution
    h = conv_3_3(x, 16, (2, 1), ((1, 1), (0, 0)), 'conv1', initializer)

    # Residual Blocks
    for block_idx, channel, freq_pooling in zip(range(1, 4), [16, 32, 64], ((1, 1), (2, 1), (2, 1))):
        h = wide_block(h, 3, channel * 5, freq_pooling, 'res' + str(block_idx), keep_prob, utt_len, initializer)

    # Normalize
    h = tf.nn.elu(normalize_4d(h, 'bn', utt_len, initializer))
    h = tf.transpose(h, (2, 0, 3, 1))

    # ELU + Linear
    h = dense_4d(h, 'pooler', tf.nn.elu, initializer)
    
    # BLSTM
    h = stacked_blstm(h, utt_len, (512, 512),'blstm', keep_prob, initializer)

    # Linear and ELU + Linear
    h = stacked_dense(h, (1024, 2042), 'classifier', utt_len, initializer)
    
    y_ = tf.identity(h, name='log_posteriors')

    # the output here is the log posterior probability
    return y_


def loss(labels, logits, utt_len):
    cross_entropy_utt = [None for i in range(utt_len.shape[0])]
    acc_utt = [None for i in range(utt_len.shape[0])]
    
    for utt in range(utt_len.shape[0]):
        cross_entropy_utt[utt] = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[utt, :utt_len[utt]], logits=logits[utt, :utt_len[utt], :]))
        _, acc_utt[utt] = tf.metrics.accuracy(labels=labels[utt, :utt_len[utt]], predictions=tf.argmax(logits[utt, :utt_len[utt], :], axis=1))
        acc_utt[utt] *= tf.cast(utt_len[utt], tf.float32)
    
    acc_sum = tf.reduce_sum(acc_utt)
    cross_entropy_sum = tf.reduce_sum(cross_entropy_utt)
    
    tf.add_to_collection('losses', cross_entropy_sum)
    tf.add_to_collection('acc', acc_sum)


def tower_loss(scope, data, labels, utt_len, batch_size, num_gpus, keep_prob, initializer):
    """
    scope: name scope for the GPU
    data: data (B x T x 240)
    labels: labels (B x T)
    """

    logits = inference(data, utt_len, batch_size, num_gpus, keep_prob, initializer)
    logits = tf.transpose(logits, [1, 0, 2])

    loss(labels, logits, utt_len)
    losses = tf.get_collection('losses', scope)

    acc = tf.get_collection('acc', scope)

    return losses, acc


# average gradients over different GPUs
def average_gradients(tower_grads):
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        grads = []

        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,axis=0)
            grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads
