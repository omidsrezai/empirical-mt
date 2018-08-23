import tensorflow as tf
from surround.smart_example import SmartConv2D as SelConv2d


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

        tf.summary.histogram('before_nonlinearity', dense)
        if act is not None:
            dense = act(dense)
            tf.summary.histogram('after_nonlinearity', dense)

        if dropout > 0:
            dense = tf.layers.dropout(dense)

        scope.reuse_variables()
        for t in ['kernel', 'bias']:
            tf.summary.histogram(t, tf.get_variable('dense/%s' % t))
            tf.summary.scalar('%s_l2_norm' % t, tf.norm(tf.get_variable('dense/%s' % t)))

    return dense


def conv2d(x, kernel_size, filters, strides, name,
           padding='same',
           act=tf.nn.relu,
           batch_norm=True,
           dropout=0.,
           kernel_l2_reg_scale=0.,
           bias_l2_reg_scale=0.,
           max_pool=None,
           max_im_outputs=64,
           k_init_uniform=True,
           activity_summary=True,
           kernel_summary=True):

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

        if max_pool is not None:
            conv = tf.layers.max_pooling2d(conv, pool_size=max_pool, strides=max_pool)

        # batch norm after pooling, reasons see this post,
        # https://www.reddit.com/r/learnmachinelearning/comments/59tuxe/do_you_do_batch_normalization_before_or_after/
        # batch norm after activation, this is debatable, see this post,
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        if batch_norm:
            conv = tf.layers.batch_normalization(conv, axis=3)

        # dropout after batch norm, see this paper,
        # https://arxiv.org/abs/1801.05134
        if dropout > 0:
            conv = tf.layers.dropout(conv, dropout)

        if activity_summary:
            tf.summary.histogram('%s_activations' % name, conv)
            tf.summary.image('%s_feature_maps_l2_norm' % name,
                             tf.norm(conv, axis=3, ord=2, keep_dims=True),
                             max_outputs=max_im_outputs)

        if kernel_summary:
            scope.reuse_variables()
            for t in ['kernel', 'bias']:
                tf.summary.histogram(t, tf.get_variable('conv2d/%s' % t))
                tf.summary.scalar('%s_l2_norm' % t, tf.norm(tf.get_variable('conv2d/%s' % t)))

    return conv


def time_map(x_timesteps, f, name):
    with tf.variable_scope('time_map_%s' % name):
        y_timesteps = []
        ts = x_timesteps[0].shape[1] if isinstance(x_timesteps, tuple) else x_timesteps.shape[1]

        for i in range(0, ts):
            y_timestep = f(*[x[:, i] for x in x_timesteps]) \
                if isinstance(x_timesteps, tuple) else f(x_timesteps[:, i])
            y_timesteps.append(y_timestep)

        y_timesteps = tf.stack(y_timesteps, axis=1)

    return y_timesteps


def chann_sel_15_by_15_conv2d(x, n_chann, k_constraint, l2_reg_scale=0., dp=0., impl=1, kernel_summary=False):
    if impl == 0:
        with tf.variable_scope('chann_sel_conv2d'):
            conv2d = SelConv2d(n_chann,
                               (15, 15),
                               activation=None,
                               use_bias=False,
                               padding="SAME",
                               kernel_constraint=k_constraint,
                               name='conv2d',
                               kernel_initializer='glorot_uniform')

            y = tf.identity(conv2d(x))  # fixes no out_bound bug

    elif impl == 1:
        y = chann_reg_conv2d(x, constraint=k_constraint,
                             filters=n_chann,
                             kernel_size=(15, 15),
                             kernel_summary=kernel_summary,
                             name='chann_sel_conv2d',
                             kernel_l2_reg_scale=l2_reg_scale)

    elif impl == 2:
        y = one_to_one_conv2d(x, constraint=k_constraint,
                              kernel_size=(15, 15),
                              kernel_summary=kernel_summary,
                              name='chann_sel_conv2d',
                              kernel_l2_reg_scale=l2_reg_scale)

    else:
        raise ValueError('Unkown chann_sel_impl %s' % impl)

    if dp > 0.:
        y = tf.layers.dropout(y, rate=dp)

    return y


def one_to_one_conv2d(x, kernel_size,
                     constraint,
                     kernel_l2_reg_scale=0.,
                     max_im_outputs=64,
                     k_init_uniform=True,
                     activity_summary=True,
                     kernel_summary=True,
                     name='chann_sel_conv2d'):
    with tf.variable_scope(name) as scope:
        n_in_chann = x.get_shape().as_list()[3]

        kernel = tf.get_variable('kernel',
                                 shape=(15, 15, n_in_chann, n_in_chann),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(k_init_uniform),
                                 dtype=tf.float32,
                                 constraint=constraint,
                                 regularizer=tf.contrib.layers.l2_regularizer(kernel_l2_reg_scale))

        kernel_selector = tf.eye(n_in_chann)

        kernel_in_mask = tf.expand_dims(tf.expand_dims(kernel_selector, axis=0), axis=1)
        kernel_in_mask = tf.tile(kernel_in_mask, list(kernel_size) + [1, 1])

        kernel_masked = kernel * kernel_in_mask
        conv = tf.nn.conv2d(x, kernel_masked,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        if activity_summary:
            tf.summary.histogram('%s_activations' % name, conv)
            tf.summary.image('%s_feature_maps_l2_norm' % name,
                             tf.norm(conv, axis=3, ord=2, keep_dims=True),
                             max_outputs=max_im_outputs)

        if kernel_summary:
            tf.summary.histogram('kernel', kernel)
            tf.summary.scalar('kernel', tf.norm(kernel, ord=2))

            with tf.name_scope('visualize_selected_kernels'):
                for i in range(n_in_chann):
                    selected_in_kernel = kernel[:, :, i, i]

                    normed = tf.abs(selected_in_kernel)
                    normed = (normed - tf.reduce_min(normed)) / \
                             (tf.reduce_max(normed) - tf.reduce_min(normed) + tf.keras.backend.epsilon())

                    tf.summary.image('out_chann%s_kernel' % i,
                                     tf.expand_dims(tf.expand_dims(normed, axis=2), axis=0),
                                     max_outputs=1)

    return conv


def chann_reg_conv2d(x, kernel_size, filters,
                     constraint,
                     kernel_l2_reg_scale=0.,
                     max_im_outputs=64,
                     k_init_uniform=True,
                     activity_summary=True,
                     kernel_summary=True,
                     name='chann_sel_conv2d'):
    with tf.variable_scope(name) as scope:
        n_in_chann = x.get_shape().as_list()[3]

        kernel = tf.get_variable('kernel',
                                 shape=(15, 15, n_in_chann, filters),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(k_init_uniform),
                                 dtype=tf.float32,
                                 constraint=constraint,
                                 regularizer=tf.contrib.layers.l2_regularizer(kernel_l2_reg_scale))


        #kernel_pooled = tf.reduce_sum(tf.abs(kernel), axis=(0, 1))
        #kernel_pooled_centered = kernel_pooled - tf.reduce_max(kernel_pooled, axis=0)
        #kernel_pooled_centered = tf.layers.batch_normalization(kernel_pooled)
        #kernel_in_gate = tf.nn.sigmoid(kernel_pooled_centered)
        #kernel_in_gate = tf.exp(tf.div(kernel_pooled_centered, 0.09))

        # kernel_pooled = _compute_conv_kernel_gradient_norm(kernel)
        kernel_pooled = tf.reduce_sum(tf.square(kernel), axis=(0, 1))
        kernel_pooled_centered = tf.divide(kernel_pooled - tf.reduce_min(kernel_pooled, axis=0),
                                           (tf.reduce_max(kernel_pooled) - tf.reduce_min(kernel_pooled)))
        kernel_pooled_centered = 1 - kernel_pooled_centered
        kernel_in_gate = tf.exp(-tf.div(tf.square(kernel_pooled_centered), 0.01))
        #kernel_in_gate = tf.nn.softmax(1 - kernel_pooled_centered, axis=0)

        '''
        if not tf.get_variable_scope().reuse:
            in_gate_l1_reg = tf.reduce_mean(kernel_in_gate)
            tf.summary.scalar('in_gate_l1_loss', in_gate_l1_reg)
            tf.losses.add_loss(in_gate_l1_reg, loss_collection= tf.GraphKeys.REGULARIZATION_LOSSES)
        '''

        kernel_in_mask = tf.expand_dims(tf.expand_dims(kernel_in_gate, axis=0), axis=1)
        kernel_in_mask = tf.tile(kernel_in_mask, list(kernel_size) + [1, 1])

        kernel_masked = kernel * kernel_in_mask
        conv = tf.nn.conv2d(x, kernel_masked,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        if activity_summary:
            tf.summary.histogram('%s_activations' % name, conv)
            tf.summary.image('%s_feature_maps_l2_norm' % name,
                             tf.norm(conv, axis=3, ord=2, keep_dims=True),
                             max_outputs=max_im_outputs)

        if kernel_summary:
            tf.summary.histogram('kernel', kernel)
            tf.summary.scalar('kernel', tf.norm(kernel, ord=2))

            with tf.name_scope('visualize_selected_kernels'):
                for i in range(filters):
                    weights = kernel_pooled[:, i]
                    selected_in_kernel = kernel[:, :, tf.argmin(weights), i]

                    normed = tf.abs(selected_in_kernel)
                    normed = (normed - tf.reduce_min(normed)) / \
                             (tf.reduce_max(normed) - tf.reduce_min(normed) + tf.keras.backend.epsilon())

                    tf.summary.image('out_chann%s_kernel' % i,
                                     tf.expand_dims(tf.expand_dims(normed, axis=2), axis=0),
                                     max_outputs=1)

            tf.summary.histogram('kernel_pooled_centered', kernel_pooled_centered)
            tf.summary.histogram('kernel_in_gates', kernel_in_gate)
            tf.summary.image('kernel_in_gates',
                            tf.expand_dims(tf.expand_dims(kernel_in_gate, axis=2), axis=0),
                            max_outputs=1)

    return conv


def _compute_conv_kernel_gradient_norm(k):
    k_trans = tf.transpose(k, perm=(3, 0, 1, 2))
    dy, dx = tf.image.image_gradients(k_trans)

    gradients = tf.square(dy) + tf.square(dx)
    gradients_per_in_chann = tf.reduce_sum(tf.square(gradients), axis=(1, 2))

    return tf.transpose(gradients_per_in_chann)
