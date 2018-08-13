import tensorflow as tf

FIXED_FRAME_SIZE = 76


class ALOV300CNNFn(object):

    def __init__(self):
        pass

    def __call__(self, features, labels, mode):
        self._draw_input_frames(features, labels)

        # input feature maps
        with tf.variable_scope('inputs'):
            h_flow = features['h_flow_f']
            h_flow_images = tf.reshape(h_flow, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])

            v_flow = features['v_flow_f']
            v_flow_images = tf.reshape(v_flow, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])

            contrast = features['contrast']
            contrast_images = tf.reshape(contrast, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])

            # tf.summary.image('optic_flow_x', h_flow_images, max_outputs=10)
            # tf.summary.image('optic_flow_y', v_flow_images, max_outputs=10)
            tf.summary.image('speed', tf.sqrt(tf.square(h_flow_images) + tf.square(v_flow_images)), max_outputs=10)
            tf.summary.image('contrast', contrast_images, max_outputs=10)
            tf.summary.histogram('optic_flow_x', h_flow)
            tf.summary.histogram('optic_flow_y', v_flow,)
            tf.summary.histogram('contrast', contrast)

            #prev_bbox_input_layer = features['prev_bbox_heat_map']
            #tf.summary.image('prev_bbox_heat_map', tf.reshape(prev_bbox_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1]), max_outputs=10)


            # show lk optic_flow
            # h_flow_lk = features['h_flow_lk']

            prev_bbox_input_layer = tf.reshape(features['prev_bbox'], [-1, 4]) / (FIXED_FRAME_SIZE - 1)

            input_feat_maps = tf.stack([h_flow, v_flow], axis=3)


        with tf.variable_scope('mt_layer'):
            mt_layer = self._conv_layer(input_feat_maps)

        '''
        with tf.variable_scope("mst_layer"):
            mst_layer = self._conv_layer(mt_layer)
        '''

        mst_layer_flat = tf.layers.flatten(mt_layer)# tf.layers.flatten(mst_layer)

        dense = tf.layers.dense(mst_layer_flat, units=1024, activation=tf.nn.tanh)
        dense = tf.layers.dense(dense, units=128, activation=tf.nn.tanh)
        dense = tf.layers.dense(dense, units=16, activation=tf.nn.tanh)

        #dense1 = tf.layers.dense(mst_layer_flat, units=1024, activation=tf.nn.sigmoid, name='dense1')
        #tf.summary.histogram('dense1_nonlinear', dense1)

        #dense2 = tf.layers.dense(dense1, units=512, activation=tf.nn.sigmoid, name='dense2')
        #tf.summary.histogram('dense2_nonlinear', dense2)

        concat = tf.concat([dense, prev_bbox_input_layer], axis=1)

        dense3 = tf.layers.dense(concat, units=8, activation=tf.nn.tanh, name='dense3')
        tf.summary.histogram('dense3_nonlinear', dense3)


        '''
        h_flow_bbox = h_flow # tf.multiply(h_flow, prev_bbox_input_layer)
        v_flow_bbox = v_flow  #tf.multiply(v_flow, prev_bbox_input_layer)

        h_dense = tf.reshape(h_flow_bbox, [-1, FIXED_FRAME_SIZE * FIXED_FRAME_SIZE])
        h_offset = tf.layers.dense(h_dense, units=1, activation=tf.nn.tanh)

        v_dense = tf.reshape(v_flow_bbox, [-1, FIXED_FRAME_SIZE * FIXED_FRAME_SIZE])
        v_offset = tf.layers.dense(v_dense, units=1, activation=tf.nn.tanh)

        offset = tf.stack([h_offset, v_offset, h_offset, v_offset], axis=1)

        output = tf.reshape(features['prev_bbox'] / (FIXED_FRAME_SIZE - 1), [-1, 1, 4]) + tf.reshape(offset, [-1, 1, 4])
        '''

        output = tf.layers.dense(dense3, units=4, activation=tf.nn.sigmoid, name='output_flat')
        self._draw_output_bbox(features['frame'], output)

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

        # evaluation mode
        eval_metrics_ops = {
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


    def _draw_input_frames(self, inputs, labels):
        with tf.variable_scope('input_frames'):
            prev_frame = inputs['prev_frame']
            prev_bbox = tf.reshape(inputs['prev_bbox'], [-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)
            frame = inputs['frame']
            ground_truth_bbox = tf.reshape(labels, shape=[-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)

            prev_frame_bbox = tf.image.draw_bounding_boxes(prev_frame, prev_bbox)
            frame_bbox = tf.image.draw_bounding_boxes(frame, ground_truth_bbox)

            tf.summary.image("frame1", prev_frame_bbox, max_outputs=10)
            tf.summary.image("frame2", frame_bbox, max_outputs=10)


    def _draw_output_bbox(self, frame, bbox):
        with tf.variable_scope('predicted_bbox'):
            bbox = tf.reshape(bbox, [-1, 1, 4])
            frame_bbox = tf.image.draw_bounding_boxes(frame, bbox)
            tf.summary.image("predicted_bbox", frame_bbox, max_outputs=10)

    def _conv_layer(self, inputs, n_filters=8):
        conv1 = tf.layers.conv2d(inputs,
                                 filters=n_filters,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.tanh)
        # pool = tf.layers.max_pooling2d(conv1,
        #                               pool_size=(3, 3),
        #                               strides=(3, 3))

        #pool_shape = tf.slice(tf.shape(pool), [1], (2,))
        #pool_images = tf.reshape(tf.reduce_mean(pool, axis=3), tf.concat([pool_shape, [1]], axis=0))
        pool = conv1

        pool_images = tf.expand_dims(tf.reduce_mean(tf.square(conv1), axis=3), axis=3)

        tf.summary.histogram("pooling", pool)
        tf.summary.image("pooling_mean", pool_images, max_outputs=10)

        return pool