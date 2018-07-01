import tensorflow as tf


class ALOV300ModelBase(object):
    def __init__(self, **kwargs):
        pass

    def _dense(self, x, units, name, act=tf.nn.tanh):

        with tf.name_scope(name):
            dense = tf.layers.dense(x,
                                    units=units,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.initializers.zeros,
                                    name=name)

            tf.summary.histogram('before_nonlinearity', dense)
            act = act(dense)
            tf.summary.histogram('after_nonlinearity', act)

        return act

    def _conv2d(self, x, kernel_size, filters, strides, name,
                padding='same',
                act=tf.nn.relu,
                batch_norm=True,
                dropout=0):

        with tf.variable_scope(name):
            conv = tf.layers.conv2d(x, kernel_size=kernel_size,
                                    strides=strides,
                                    filters=filters,
                                    padding=padding,
                                    activation=act,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    bias_initializer=tf.initializers.zeros,
                                    name=name)

            if batch_norm:
                conv = tf.layers.batch_normalization(conv, axis=3)

            if dropout > 0:
                conv = tf.layers.dropout(conv, dropout)

            tf.summary.histogram('%s_activations' % name, conv)
            tf.summary.image('%s_feature_maps_l2_norm' % name,
                             tf.norm(conv, axis=3, ord=2, keep_dims=True),
                             max_outputs=64)

        return conv
