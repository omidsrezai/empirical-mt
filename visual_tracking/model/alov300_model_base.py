from os import path

import tensorflow as tf

LEARNING_RATE = 0.0001 # was 0.0001


class ALOV300ModelBase(object):

    def __init__(self, max_image_outputs=64):
        self.max_im_outputs = max_image_outputs

    def _compile(self, mode, frame1, frame2, prev_box, labels, pbbox, p_delta):
        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pbbox)

        # TRAIN mode
        with tf.name_scope('draw_bboxes'):
            with tf.name_scope('frame1'):
                self._summary_images_with_bbox(frame1, prev_box, name='frame1_box')

            with tf.name_scope('frame2'):
                self._summary_images_with_bbox(frame2, labels, name='frame2_box')
                self._summary_images_with_bbox(frame2, pbbox, name='frame2_predicted_box')

        self._predicted_delta_summary(prev_bbox=prev_box, p_bbox=pbbox, t_bbox=labels)

        bbox_loss = tf.losses.absolute_difference(labels=labels - prev_box, predictions=p_delta)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.minimize(
                loss=bbox_loss + tf.losses.get_regularization_loss(),
                global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, train_op=train_op)

        # EVAL mode
        eval_metrics_ops = self._get_eval_metrics_ops(predictions=pbbox - prev_box,
                                                      labels=labels - prev_box)
        return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, eval_metric_ops=eval_metrics_ops)

    def _dense(self, x, units, name,
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

    def _conv2d(self, x, kernel_size, filters, strides, scope_name,
                padding='same',
                conv_name='conv2d',
                act=tf.nn.relu,
                batch_norm=True,
                dropout=0.,
                kernel_l2_reg_scale=0.,
                bias_l2_reg_scale=0.,
                reuse=None):

        with tf.variable_scope(scope_name) as scope:
            conv = tf.layers.conv2d(x,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    filters=filters,
                                    padding=padding,
                                    activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    bias_initializer=tf.initializers.zeros,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_l2_reg_scale),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(bias_l2_reg_scale),
                                    reuse=reuse,
                                    name=conv_name)

            if batch_norm:
                conv = tf.layers.batch_normalization(conv, axis=3, reuse=reuse)

            if act is not None:
                conv = act(conv)

            if dropout > 0:
                conv = tf.layers.dropout(conv, dropout)

            tf.summary.histogram('%s_activations' % scope_name, conv)
            tf.summary.image('%s_feature_maps_l2_norm' % scope_name,
                             tf.norm(conv, axis=3, ord=2, keep_dims=True),
                             max_outputs=self.max_im_outputs)

            if reuse is None:
                scope.reuse_variables()
                for t in ['kernel', 'bias']:
                    tf.summary.histogram(t, tf.get_variable('conv2d/%s' % t))
                    tf.summary.scalar('%s_l2_norm' % t, tf.norm(tf.get_variable('conv2d/%s' % t)))

        return conv

    def _get_eval_metrics_ops(self, labels, predictions):
        _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']
        pred_mean = tf.reduce_mean(predictions, axis=0)

        metrics_ops = {
            'l1_loss': tf.metrics.mean_absolute_error(labels=labels, predictions=predictions),
            'l2_loss': tf.metrics.mean_squared_error(labels=labels, predictions=predictions),
        }

        for i, v in enumerate(_dim_ids):
            metrics_ops['%s_mean' % v] = tf.metrics.mean(predictions[:, i])
            metrics_ops['%s_var' % v] = tf.metrics.mean(tf.square(predictions[:, i] - pred_mean[i]))
            metrics_ops['%s_covar' % v] = tf.contrib.metrics.streaming_covariance(predictions=predictions[:, i],
                                                                                  labels=labels[:, i])
            metrics_ops['%s_corr' % v] = tf.contrib.metrics.streaming_pearson_correlation(predictions=predictions[:, i],
                                                                                          labels=labels[:, i])

        return metrics_ops

    def _summary_images_with_bbox(self, images, boxes, name):
        boxes_3d = tf.stack([boxes, boxes, boxes], axis=1)
        annotated_im = tf.image.draw_bounding_boxes(images, boxes_3d)
        tf.summary.image(name, annotated_im, max_outputs=self.max_im_outputs)

    def _predicted_delta_summary(self, prev_bbox, p_bbox, t_bbox):
        _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']

        with tf.name_scope('prediction_metrics/delta'):
            p_delta = p_bbox - prev_bbox
            t_delta = t_bbox - prev_bbox

            t_mean, t_var = tf.nn.moments(t_delta, axes=0)
            p_mean, p_var = tf.nn.moments(p_delta, axes=0)

            covar = tf.reduce_mean(tf.multiply(p_delta, t_delta), axis=0) - tf.multiply(t_mean, p_mean)
            corr = tf.div(covar, tf.sqrt(tf.multiply(t_var, p_var)))

            for i, v in enumerate(_dim_ids):
                tf.summary.histogram('t_%s_delta' % v, t_delta[:, i])
                tf.summary.histogram('p_%s_delta' % v, p_delta[:, i])

                tf.summary.scalar('t_%s_delta_mean' % v, t_mean[i])
                tf.summary.scalar('p_%s_delta_mean' % v, p_mean[i])

                tf.summary.scalar('t_%s_delta_var' % v, t_var[i])
                tf.summary.scalar('p_%s_delta_var' % v, p_var[i])

                tf.summary.scalar('corvar_delta_%s' % v, covar[i])
                tf.summary.scalar('corr_delta_%s' % v, corr[i])

            tf.summary.scalar('l1_loss', tf.norm(p_bbox - t_bbox, ord=1))
            tf.summary.scalar('l2_loss', tf.norm(p_bbox - t_bbox, ord=2))
