import tensorflow as tf

FIXED_FRAME_SIZE = 76


class ALOV300GOTURNFn(object):

    def __init__(self):
        pass

    def __call__(self, features, labels, mode):
        self._draw_input_frames(features, labels)

        # input feature maps
        with tf.variable_scope('inputs'):
            search_patch = tf.reshape(features['search_patch'], [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            object_patch = tf.reshape(features['object_patch'], [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])

        with tf.variable_scope('search_patch_path'):
            search_patch_features = self._conv_layers(search_patch)
            search_patch_features = tf.layers.flatten(search_patch_features)

        with tf.variable_scope("object_patch_path"):
            object_patch_features = self._conv_layers(object_patch)
            object_patch_features = tf.layers.flatten(object_patch_features)

        concat = tf.concat([search_patch_features, object_patch_features], axis=1)

        dense1 = tf.layers.dense(concat, units=1024, activation=tf.nn.sigmoid, name='dense1')
        tf.summary.histogram('dense1_nonlinear', dense1)

        dense2 = tf.layers.dense(dense1, units=512, activation=tf.nn.sigmoid, name='dense2')
        tf.summary.histogram('dense2_nonlinear', dense2)

        # concat = tf.concat([dense2, prev_bbox_input_layer], axis=1)

        dense3 = tf.layers.dense(dense2, units=64, activation=tf.nn.sigmoid, name='dense3')
        tf.summary.histogram('dense3_nonlinear', dense3)

        dense4 = tf.layers.dense(dense3, units=16, activation=tf.nn.sigmoid, name='dense4')
        tf.summary.histogram('dense4_nonlinear', dense4)

        output = tf.layers.dense(dense4, units=4, activation=tf.nn.sigmoid, name='output_flat')

        # a nonlinearity
        # output = tf.where(tf.greater(output, 1.), tf.ones_like(output), output)
        # output = tf.where(tf.less(output, 0.), tf.zeros_like(output), output)

        tf.summary.histogram('output', output)

        self._draw_output_bbox(features['search_patch'], output)

        output = tf.reshape(output, [-1, 2, 2])

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            # rescale it to coordinates
            predictions = output * (FIXED_FRAME_SIZE - 1)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        with tf.variable_scope('ground_truth_bbox'):
            labels_rescaled = labels / (FIXED_FRAME_SIZE - 1)

        with tf.variable_scope('metrics'):
            p_offsets_in_pixels = features['prev_bbox'] - (output * (FIXED_FRAME_SIZE - 1))
            p_offsets_mean = tf.reduce_mean(p_offsets_in_pixels)
            p_offsets_var = tf.reduce_mean(tf.square(p_offsets_in_pixels - p_offsets_mean))
            tf.summary.scalar('predicted_offset_mean', p_offsets_mean)
            tf.summary.scalar('predicted_offset_var', p_offsets_var)

            t_offsets_in_pixels = features['prev_bbox'] - labels
            t_offsets_mean = tf.reduce_mean(t_offsets_in_pixels)
            t_offset_var = tf.reduce_mean(tf.square(t_offsets_in_pixels - t_offsets_mean))

            offsets_covar = tf.reduce_mean((t_offsets_in_pixels - t_offsets_mean) *
                                           (p_offsets_in_pixels - p_offsets_mean))
            tf.summary.scalar('true_offset_mean', t_offsets_mean)
            tf.summary.scalar('true_offset_var', t_offset_var)
            tf.summary.scalar('offsets_covariance', offsets_covar)

        # l1 loss
        loss = tf.losses.absolute_difference(labels=labels_rescaled, predictions=output)

        # TRAIN mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def _draw_input_frames(self, inputs, labels):
        with tf.variable_scope('input_frames'):
            prev_frame = inputs['object_patch']
            prev_bbox = tf.reshape(inputs['prev_bbox'], [-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)
            frame = inputs['search_patch']
            ground_truth_bbox = tf.reshape(labels, shape=[-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)

            prev_frame_bbox = tf.image.draw_bounding_boxes(prev_frame, prev_bbox)
            frame_bbox = tf.image.draw_bounding_boxes(frame, ground_truth_bbox)

            tf.summary.image("frame1", prev_frame_bbox)
            tf.summary.image("frame2", frame_bbox)


    def _draw_output_bbox(self, frame, bbox):
        with tf.variable_scope('predicted_bbox'):
            bbox = tf.reshape(bbox, [-1, 1, 4])
            frame_bbox = tf.image.draw_bounding_boxes(frame, bbox)
            tf.summary.image("predicted_bbox", frame_bbox)

    def _conv_layers(self, inputs, n_filters=64):
        conv1 = tf.layers.conv2d(inputs,
                                 filters=n_filters,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1,
                                       pool_size=(3, 3),
                                       strides=(3, 3))

        conv2 = tf.layers.conv2d(pool1,
                                 filters=n_filters,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2,
                                        pool_size=(3, 3),
                                        strides=(3, 3))

        #pool_shape = tf.slice(tf.shape(pool), [1], (2,))
        #pool_images = tf.reshape(tf.reduce_mean(pool, axis=3), tf.concat([pool_shape, [1]], axis=0))
        pool_images = tf.expand_dims(tf.reduce_mean(pool2, axis=3), axis=3)

        tf.summary.histogram("pooling", pool2)
        tf.summary.image("pooling_mean", pool_images)

        return pool2