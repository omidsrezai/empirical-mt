import tensorflow as tf


class ALOV300ModelBase(object):

    def __init__(self, max_image_outputs=64,
                 lr=0.0001,
                 loss='l1',
                 lr_decay_steps=50,
                 lr_decay_rate=0.99):
        self.max_im_outputs = max_image_outputs
        self.learing_rate = lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.loss = loss

    def _compile(self,
                 mode,
                 frame1,
                 frame2,
                 prev_box,
                 ground_truth_box,
                 pred_box,
                 y_hat,
                 y):

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_box)

        # TRAIN mode
        self._train_prediction_summary(prev_bbox=prev_box,
                                       p_bbox=pred_box,
                                       t_bbox=ground_truth_box,
                                       frame1=frame1,
                                       frame2=frame2)

        if self.loss == 'l1':
            bbox_loss = tf.losses.absolute_difference(labels=y, predictions=y_hat)
        elif self.loss == 'l2':
            bbox_loss = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
        else:
            raise ValueError('%s loss undefined' % self.loss)

        total_loss = bbox_loss + tf.losses.get_regularization_loss()

        learning_rate = tf.train.exponential_decay(learning_rate=self.learing_rate,
                                                   global_step=tf.train.get_global_step(),
                                                   decay_steps=self.lr_decay_steps,
                                                   decay_rate=self.lr_decay_rate)
        tf.summary.scalar('learning_rate', learning_rate)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss=total_loss,
                global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

        # EVAL mode
        eval_metrics_ops = self._get_eval_metrics_ops(p_delta=pred_box - prev_box,
                                                      t_delta=ground_truth_box - prev_box)

        return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, eval_metric_ops=eval_metrics_ops)

    def _get_eval_metrics_ops(self, t_delta, p_delta):
        _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']
        pred_mean = tf.reduce_mean(p_delta, axis=0)

        metrics_ops = {
            'l1_loss': tf.metrics.mean_absolute_error(labels=t_delta, predictions=p_delta),
            'l2_loss': tf.metrics.mean_squared_error(labels=t_delta, predictions=p_delta),
            'corr_concat': tf.contrib.metrics\
                .streaming_pearson_correlation(predictions=tf.reshape(t_delta, [-1]),
                                               labels=tf.reshape(p_delta, [-1]))
        }

        for i, v in enumerate(_dim_ids):
            metrics_ops['%s_mean' % v] = tf.metrics.mean(p_delta[:, i])
            metrics_ops['%s_var' % v] = tf.metrics.mean(tf.square(p_delta[:, i] - pred_mean[i]))
            metrics_ops['%s_covar' % v] = tf.contrib.metrics\
                .streaming_covariance(predictions=p_delta[:, i], labels=t_delta[:, i])
            metrics_ops['%s_corr' % v] = tf.contrib.metrics\
                .streaming_pearson_correlation(predictions=p_delta[:, i], labels=t_delta[:, i])

        return metrics_ops

    def _summary_images_with_bbox(self, images, boxes, name):
        boxes_3d = tf.stack([boxes, boxes, boxes], axis=1)
        annotated_im = tf.image.draw_bounding_boxes(images, boxes_3d)
        tf.summary.image(name, annotated_im, max_outputs=self.max_im_outputs)

    def _train_prediction_summary(self, prev_bbox, p_bbox, t_bbox, frame1, frame2):
        with tf.name_scope('train_prediction_summary'):

            # draw predicted bounding box
            with tf.name_scope('draw_bboxes'):
                with tf.name_scope('frame1'):
                    self._summary_images_with_bbox(frame1, prev_bbox, name='frame1_box')

                with tf.name_scope('frame2'):
                    self._summary_images_with_bbox(frame2, t_bbox, name='frame2_box')
                    self._summary_images_with_bbox(frame2, p_bbox, name='frame2_predicted_box')

            # plot predicted offsets
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

            # plot losses
            with tf.name_scope('losses'):
                tf.summary.scalar('l1_loss', tf.reduce_mean(tf.abs(p_bbox - t_bbox)))
                tf.summary.scalar('l2_loss', tf.reduce_mean(tf.square(p_bbox - t_bbox)))
