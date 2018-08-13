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

    def _compile(self, mode, frame1, frame2, bbox_frame1, bbox_true, bbox_pred, y_pred, y_true):
        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=bbox_pred)

        # TRAIN mode
        self._train_prediction_summary(bbox_frame1=bbox_frame1,
                                       bbox_pred=bbox_pred,
                                       bbox_true=bbox_true,
                                       frame1=frame1,
                                       frame2=frame2)

        if self.loss == 'l1':
            bbox_loss = tf.losses.absolute_difference(labels=y_true, predictions=y_pred)
        elif self.loss == 'l2':
            bbox_loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
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
        eval_metrics_ops = self._get_eval_metrics_ops(y_true=y_true, y_pred=y_pred,
                                                      delta_pred=bbox_pred - bbox_frame1,
                                                      delta_true=bbox_true - bbox_frame1)

        return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, eval_metric_ops=eval_metrics_ops)

    def _get_eval_metrics_ops(self, y_true, y_pred, delta_true, delta_pred):
        _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']

        # computes IoU
        y_min_t, x_min_t, y_max_t, x_max_t = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
        y_min_p, x_min_p, y_max_p, x_max_p = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

        bbox_area_true = tf.multiply(y_max_t - y_min_t, x_max_t - x_min_t)
        bbox_area_pred = tf.multiply(y_max_p - y_min_p, x_max_p - x_min_p)

        intersec = tf.multiply(tf.minimum(y_max_t, y_max_p) - tf.maximum(y_min_t, y_min_p),
                               tf.minimum(x_max_t, x_max_p) - tf.maximum(x_min_t, x_min_p))
        union = bbox_area_true + bbox_area_pred - intersec
        iou = tf.divide(intersec, union + tf.keras.backend.epsilon())

        metrics_ops = {
            'mean_iou': tf.metrics.mean(iou),
            'l1_loss': tf.metrics.mean_absolute_error(labels=delta_true, predictions=delta_pred),
            'l2_loss': tf.metrics.mean_squared_error(labels=delta_true, predictions=delta_pred),
            'corr_concat': tf.contrib.metrics\
                .streaming_pearson_correlation(predictions=tf.reshape(delta_true, [-1]),
                                               labels=tf.reshape(delta_pred, [-1]))
        }

        # computes correlation for 4 predicted numbers
        mean_pred = tf.reduce_mean(delta_pred, axis=0)
        for i, v in enumerate(_dim_ids):
            metrics_ops['%s_mean' % v] = tf.metrics.mean(delta_pred[:, i])
            metrics_ops['%s_var' % v] = tf.metrics.mean(tf.square(delta_pred[:, i] - mean_pred[i]))
            metrics_ops['%s_covar' % v] = tf.contrib.metrics\
                .streaming_covariance(predictions=delta_pred[:, i], labels=delta_true[:, i])
            metrics_ops['%s_corr' % v] = tf.contrib.metrics\
                .streaming_pearson_correlation(predictions=delta_pred[:, i], labels=delta_true[:, i])

        return metrics_ops

    def _summary_images_with_bbox(self, images, bboxes, name):
        annotated_im = tf.image.draw_bounding_boxes(images, bboxes)
        tf.summary.image(name, annotated_im, max_outputs=self.max_im_outputs)

    def _train_prediction_summary(self, bbox_frame1, bbox_pred, bbox_true, frame1, frame2):
        with tf.name_scope('train_prediction_summary'):

            # draw predicted bounding box
            with tf.name_scope('draw_bboxes'):
                with tf.name_scope('frame1'):
                    self._summary_images_with_bbox(frame1, tf.expand_dims(bbox_frame1, axis=1), name='frame1_box')

                with tf.name_scope('frame2'):
                    self._summary_images_with_bbox(frame2, tf.stack([bbox_pred, bbox_true], axis=1), name='frame2_box')

            # plot predicted offsets
            _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']

            with tf.name_scope('prediction_metrics/delta'):
                p_delta = bbox_pred - bbox_frame1
                t_delta = bbox_true - bbox_frame1

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
                tf.summary.scalar('l1_loss', tf.reduce_mean(tf.abs(bbox_pred - bbox_true)))
                tf.summary.scalar('l2_loss', tf.reduce_mean(tf.square(bbox_pred - bbox_true)))
