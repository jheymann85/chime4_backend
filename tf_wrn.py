import inspect
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

model_path = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

# Helper functions

### Normalization ###
def normalize_3d(x, gamma, beta):
    with tf.name_scope('normalize', values=[x, gamma, beta]):
        mean, var = tf.nn.moments(
            x, axes=(0,), keep_dims=True,
            name='moments_for_axis_{}'.format('_'.join(map(str, (0,))))
        )
        x = tf.subtract(x, mean, name='normalize_mean')
        x = tf.truediv(x, (tf.sqrt(var) + 1e-8), name='normalize_variance')
        return gamma[None, None, :] * x + beta[None, None, :]

def normalize_4d(x, scope, initializer):
    with tf.variable_scope(scope, values=[x]):
        channels = x.get_shape().as_list()[-1]
        gamma = tf.get_variable(
            'gamma', channels,
            initializer=tf.constant_initializer(
                value=initializer[scope + '/gamma'])
        )
        beta = tf.get_variable(
            'beta', channels,
            initializer=tf.constant_initializer(
                value=initializer[scope + '/beta'])
        )
    with tf.name_scope(scope.split('/')[-1], values=[x, gamma, beta]):
        mean, var = tf.nn.moments(
            x, axes=(1, 2), keep_dims=True,
            name='moments_for_axis_{}'.format('_'.join(map(str, (1, 2))))
        )
        x = tf.subtract(x, mean, name='normalize_mean')
        x = tf.truediv(x, (tf.sqrt(var) + 1e-5), name='normalize_variance')
        return gamma * x + beta

### Conv ###
def conv_3_3(
        x, out_channels, strides, pads, scope, initializer
    ):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        if np.sum(pads):
            x = tf.pad(x, ((0, 0), *pads, (0, 0)))
            cut = lambda x: x[:, :-1, :, :]
        else:
            cut = lambda x: x

        with tf.variable_scope(scope, values=[x]):
            _shape = [3, 3, x.get_shape().as_list()[3], out_channels]
            filters = tf.get_variable(
                name='W',
                shape=_shape,
                initializer=tf.constant_initializer(
                    value=initializer[scope + '/W'][:].transpose(2, 3, 1, 0)
                )
            )

        z = tf.nn.conv2d(
            x, filters, strides=(1, *strides, 1), padding='SAME',
            data_format='NHWC'
        )

        z = cut(z)

    return z


def conv_1_1(
        x, out_channels, strides, scope, initializer
    ):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        ch_idx = 3
        strides = (1, *strides, 1)

        with tf.variable_scope(scope, values=[x]):
            _shape = [1, 1, x.get_shape().as_list()[ch_idx], out_channels]
            filters = tf.get_variable(
                name='W',
                shape=_shape,
                initializer=tf.constant_initializer(
                    value=initializer[scope + '/W'][:].transpose(2, 3, 1, 0)
                )
            )

        z = tf.nn.conv2d(
            x, filters, strides=strides, padding='SAME',
            data_format='NHWC'
        )

    return z


def wide_layer(
        x, out_channels, strides, scope, initializer
    ):
    with tf.name_scope(scope, values=[x]):
        in_channels = x.get_shape().as_list()[-1]
        x_norm = tf.nn.elu(normalize_4d(x, scope + '/bn1', initializer))

        if strides[0] > 1:
            pads = ((1, 1), (0, 0))
        else:
            pads = ((0, 0), (0, 0))

        if in_channels != out_channels:
            ident = conv_1_1(
                x_norm, out_channels, strides, scope + '/conv_ident',
                initializer
            )
        else:
            ident = x

        h1 = conv_3_3(
            x_norm, out_channels, strides, pads, scope + '/conv1',
            initializer
        )
        h1 = tf.nn.elu(normalize_4d(h1, scope + '/bn2', initializer))
        h2 = conv_3_3(
            h1, out_channels, (1, 1), ((0, 0), (0, 0)), scope + '/conv2',
            initializer
        )

        return h2 + ident


def wide_block(
        x, layer, output_channels, strides, scope,
        initializer
    ):
    h = wide_layer(
        x, output_channels, strides, scope + '/a', initializer
    )
    for l_idx in range(1, layer):
        h = wide_layer(
            h, output_channels, (1, 1), scope + '/b' + str(l_idx), initializer
        )
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


def lstm(x, sequence_length, units, scope, initializer, reverse):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        if reverse:
            x = tf.reverse_sequence(x, sequence_length, seq_axis=0, batch_axis=1)
        in_size = x.get_shape().as_list()[-1]
        with tf.variable_scope(scope):
            gamma = tf.get_variable(
                'gamma', shape=(4*units,),
                initializer=tf.constant_initializer(
                    value=_stride(initializer[scope + '/gamma'][:]))
            )
            beta = tf.get_variable(
                'beta', shape=(4*units,),
                initializer=tf.constant_initializer(
                    value=_stride(initializer[scope + '/beta'][:]))
            )
            W_x = tf.get_variable(
                'W_x', shape=(in_size, 4*units),
                initializer=tf.constant_initializer(
                    value=_stride(initializer[scope + '/W_x'][:]))
            )
            W_h = tf.get_variable(
                'W_h', shape=(units, 4*units),
                initializer=tf.constant_initializer(
                    value=_stride(initializer[scope + '/W_h'][:]))
            )
        with tf.name_scope('input_transform'):
            x = normalize_3d(tf.einsum('tbf,fu->tbu', x, W_x), gamma, beta)
        cell = rnn.CompiledWrapper(LSTMCell(units, W_h))
        h, _ = tf.nn.dynamic_rnn(
            cell, x, time_major=True, dtype=x.dtype,
            sequence_length=sequence_length
        )
        if reverse:
            h = tf.reverse_sequence(
                h, sequence_length, seq_axis=0, batch_axis=1
            )
    return h


def blstm(x, sequence_length, stack_output, units, scope, initializer):
    with tf.name_scope(scope):
        fw_scope = scope + '/lstm_fw'
        bw_scope = scope + '/lstm_bw'
        h_fw = lstm(x, sequence_length, units, fw_scope, initializer, False)
        h_bw = lstm(x, sequence_length, units, bw_scope, initializer, True)
        if stack_output:
            h = tf.concat([h_fw, h_bw], axis=2)
        else:
            h = h_fw + h_bw
        return h


def stacked_blstm(x, sequence_length, units, scope, initializer):
    h = x
    for idx, num_units in enumerate(units):
        _scope = scope + '/' + str(idx)
        h = blstm(
            h, sequence_length, idx == len(units)-1, num_units,
            _scope, initializer
        )
    return h


### Dense ###
def dense(
        x, units, scope, initializer, normalize, activation,
        transformation='tbf,fu->tbu'
    ):
    with tf.name_scope(scope.split('/')[-1], values=[x]):
        in_size = x.get_shape().as_list()[-1]
        with tf.variable_scope(scope):
            if normalize:
                gamma = tf.get_variable(
                    'gamma', shape=(units,),
                    initializer=tf.constant_initializer(
                        value=(initializer[scope + '/gamma'][:]))
                )
                beta = tf.get_variable(
                    'beta', shape=(units,),
                    initializer=tf.constant_initializer(
                        value=(initializer[scope + '/beta'][:]))
                )
            else:
                b_val = initializer[scope + '/b'][:]
                b = tf.get_variable(
                    'b', shape=b_val.shape,
                    initializer=tf.constant_initializer(value=b_val)
                )
            W = tf.get_variable(
                'W', shape=(in_size, units),
                initializer=tf.constant_initializer(
                    value=(initializer[scope + '/W'][:]))
            )
        with tf.name_scope('input_transform'):
            x = tf.einsum(transformation, x, W)
        if normalize:
            x = normalize_3d(x, gamma, beta)
        else:
            x += b[None, None, :]
    return activation(x)


def stacked_dense(x, units, scope, initializer):
    with tf.name_scope(scope):
        h = x
        for l_idx, num_units in enumerate(units):
            if  l_idx == len(units) - 1:
                activation = lambda x: x
            else:
                activation = tf.nn.elu
            h = dense(
                h, num_units,
                scope + '/' + str(l_idx),
                initializer,
                l_idx != len(units) - 1, activation
            )
    return h


def build_resnet(x: tf.Tensor, input_channels=3):
    assert x.shape.ndims == 2

    probes = dict()

    hdf5_file = os.path.join(
        model_path, 'assets', 'network', 'best.nnet'
    )

    initializer = dict()
    param_data = h5py.File(hdf5_file)
    data_stack = [(param_data, '/')]
    while len(data_stack):
        d, path = data_stack.pop()
        if hasattr(d, 'shape'):
            initializer[path[1:-1]] = np.array(d)
        if hasattr(d, 'keys'):
            for key in d.keys():
                data_stack.append((d[key], path + key + '/'))

    with tf.name_scope('to_bhwc'):
        _feature_dim = x.get_shape().as_list()[1]
        x = tf.reshape(x, (-1, input_channels, _feature_dim // input_channels))
        x = tf.transpose(x, (2, 0, 1))[None, ...]

    # Input cov
    h = conv_3_3(
        x, 16, (2, 1), ((1, 1), (0, 0)), 'conv1', initializer
    )
    probes['conv1'] = tf.transpose(h, (0, 3, 1, 2))
    # Blocks
    for bloock_idx, channel, freq_pooling in zip(
        range(1, 4), [16, 32, 64], ((1, 1), (2, 1), (2, 1))
    ):
        h = wide_block(
            h, 3, channel * 5, freq_pooling, 'res' + str(bloock_idx),
            initializer
        )
        probes['res' + str(bloock_idx)] = tf.transpose(h, (0, 3, 1, 2))

    # Normalize
    h = tf.nn.elu(
        normalize_4d(h, 'bn', initializer)
    )
    probes['after_bn'] = tf.transpose(h, (0, 3, 1, 2))
    _height = h.get_shape().as_list()[1]
    h = tf.transpose(h, (2, 0, 3, 1))
    h = dense(
        h, _height, 'pooler', initializer, False, tf.nn.elu,
        transformation='tbcf,df->tbcd'
    )[..., 0]
    probes['after_cnn'] = h

    h = stacked_blstm(
        h, tf.expand_dims(tf.shape(h)[0], axis=0), (512, 512),
        'blstm', initializer
    )
    probes['after_blstm'] = h

    h = stacked_dense(
        h, (1024, 2042), 'classifier', initializer
    )
    h = tf.identity(h, name='state_posteriors')
    probes['after_dense'] = h

    pdf_counts = tf.constant(initializer['pdf_counts'][:])
    normalizer = tf.log(pdf_counts / tf.reduce_sum(pdf_counts))
    probes['log_likelihoods'] = (h - normalizer)[:, 0, :]
    return probes
