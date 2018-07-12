import tensorflow as tf


def dense(x, units, name,
          act=tf.nn.tanh,
          batch_norm=False,
          dropout=0,
          kernel_l2_reg_scale=0.,
          bias_l2_reg_scale=0.):
    with tf.variable_scope(name) as scope:
        dense = tf.layers.dense(x,
                                units=units,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.initializers.zeros,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_l2_reg_scale),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(bias_l2_reg_scale))

        if batch_norm:
            dense = tf.layers.batch_normalization(dense)

        if dropout > 0:
            dense = tf.layers.dropout(dense)

        tf.summary.histogram('before_nonlinearity', dense)
        if act is not None:
            act = act(dense)
            tf.summary.histogram('after_nonlinearity', act)
        else:
            act = dense

        scope.reuse_variables()
        for t in ['kernel', 'bias']:
            tf.summary.histogram(t, tf.get_variable('dense/%s' % t))
            tf.summary.scalar('%s_l2_norm' % t, tf.norm(tf.get_variable('dense/%s' % t)))

    return act


def conv2d(x, kernel_size, filters, strides, name,
           padding='same',
           act=tf.nn.relu,
           batch_norm=True,
           dropout=0.,
           kernel_l2_reg_scale=0.,
           bias_l2_reg_scale=0.,
           max_pool=None,
           max_im_outputs=64,
           k_init_uniform=True):

    with tf.variable_scope(name) as scope:
        conv = tf.layers.conv2d(x,
                                kernel_size=kernel_size,
                                strides=strides,
                                filters=filters,
                                padding=padding,
                                activation=act,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(k_init_uniform),
                                bias_initializer=tf.initializers.zeros,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_l2_reg_scale),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(bias_l2_reg_scale))

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, axis=3)

        if max_pool is not None:
            conv = tf.layers.max_pooling2d(conv, pool_size=max_pool, strides=max_pool)

        if dropout > 0:
            conv = tf.layers.dropout(conv, dropout)

        tf.summary.histogram('%s_activations' % name, conv)
        tf.summary.image('%s_feature_maps_l2_norm' % name,
                         tf.norm(conv, axis=3, ord=2, keep_dims=True),
                         max_outputs=max_im_outputs)

        scope.reuse_variables()
        for t in ['kernel', 'bias']:
            tf.summary.histogram(t, tf.get_variable('conv2d/%s' % t))
            tf.summary.scalar('%s_l2_norm' % t, tf.norm(tf.get_variable('conv2d/%s' % t)))

    return conv