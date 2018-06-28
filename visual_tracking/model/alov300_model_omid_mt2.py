import pickle

import tensorflow as tf
from keras import  backend as K
import numpy as np

# link keras session and tensorflow session
import MT
from tuning import SpeedTuning, DirectionTuning

MAX_IMG_OUTPUTS = 64

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTTracker(object):

    def __init__(self, mt_params_path, speed_scaler):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = 64
        self.speed_scaler = speed_scaler

    def __call__(self, features, labels, mode):

        with tf.variable_scope('speed_input'):
            speed_input = tf.identity(features['speed'])
            tf.summary.histogram('speed_input', speed_input)
            speed_image = tf.reshape(speed_input, [-1, 76, 76, 1])
            tf.summary.image('speed', speed_image, max_outputs=MAX_IMG_OUTPUTS)

        with tf.variable_scope('direction_input'):
            direction_input = tf.identity(features['direction'])
            tf.summary.histogram('direction_input', direction_input)
            direction_image = tf.reshape(direction_input, [-1, 76, 76, 1])
            tf.summary.image('direction', direction_image, max_outputs=MAX_IMG_OUTPUTS)

        with tf.variable_scope('mt_tunning_layers'):
            speed_tun_layer = SpeedTuning(64, self.mt_params, self.speed_scaler, name='speed_tunning')
            direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')

            speed_tun_layer = speed_tun_layer([speed_input, tf.zeros_like(speed_input)])
            direction_tun_layer = direction_tun_layer(direction_input)
            mt_activity = tf.multiply(speed_tun_layer, direction_tun_layer, name='mt_act_tensor')

            # visualize speeding tuning and direction tuning
            tf.summary.histogram("speed_tuning", speed_tun_layer)
            speed_tun_inf_norm = tf.norm(speed_tun_layer, ord=np.inf, axis=3, keep_dims=True)
            speed_tun_l2 = tf.norm(speed_tun_layer, ord=2, axis=3, keep_dims=True)
            tf.summary.image('speed_tun_inf_norm', speed_tun_inf_norm, max_outputs=MAX_IMG_OUTPUTS)
            tf.summary.image('speed_tun_l2_norm', speed_tun_l2, max_outputs=MAX_IMG_OUTPUTS)

            tf.summary.histogram("direction_tuning", direction_tun_layer)
            tf.summary.histogram('activations', mt_activity)
            mt_act_l2 = tf.norm(mt_activity, ord=2, axis=3, keep_dims=True)
            mt_act_inf = tf.norm(mt_activity, ord=np.inf, axis=3, keep_dims=True)
            tf.summary.image('mt_act_2_norm', mt_act_l2, max_outputs=MAX_IMG_OUTPUTS)
            tf.summary.image('mt_act_inf_norm', mt_act_inf, max_outputs=MAX_IMG_OUTPUTS)

        conv = tf.layers.conv2d(mt_activity,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                filters=64,
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.initializers.zeros,
                                name='conv1')
        conv = tf.layers.conv2d(conv, kernel_size=(3, 3),
                                strides=(1, 1),
                                filters=2,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.initializers.zeros,
                                name='conv2')

        tf.summary.histogram('conv2', conv)

        pool = tf.layers.average_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))
        tf.summary.image('pool_l2_norm', tf.norm(pool, ord=2, axis=3, keep_dims=True), max_outputs=MAX_IMG_OUTPUTS)
        #tf.summary.image('pool_l1_norm', tf.norm(pool, ord=1, axis=3, keep_dims=True), max_outputs=MAX_IMG_OUTPUTS)
        #tf.summary.image('pool_inf_norm', tf.norm(pool, ord=np.inf, axis=3, keep_dims=True), max_outputs=MAX_IMG_OUTPUTS)
        tf.summary.histogram('pool', pool)

        with tf.variable_scope('conv_supervise'):
            flow_input_x = tf.multiply(speed_input, tf.cos(direction_input))
            flow_input_y = tf.multiply(speed_input, tf.sin(direction_input))

            tf.summary.image('flow_x_input_l1', tf.expand_dims(tf.abs(flow_input_x), axis=3))
            tf.summary.image('flow_y_input_l1', tf.expand_dims(tf.abs(flow_input_y), axis=3))

            conv_loss = tf.losses.mean_squared_error(labels=tf.stack([flow_input_y, flow_input_x], axis=3),
                                                     predictions=conv)
            tf.summary.scalar('conv_loss', conv_loss)

        pool_flatten = tf.layers.flatten(pool)
        pool_flatten = tf.layers.dropout(pool_flatten, 0.2)

        dense = self._dense_layer(pool_flatten, units=1024, name='dense1')
        dense = self._dense_layer(dense, units=256, name='dense2')
        out = self._dense_layer(dense, units=2, name='dense3', act=tf.nn.sigmoid)

        with tf.variable_scope('previous_frame'):
            self._bounding_box_summary(features['frame1'], features['box'], name='frame1_box')

        with tf.variable_scope('frame'):
            self._bounding_box_summary(features['frame2'], labels, name='frame2_box')

        # with tf.variable_scope('prediction'):
        #    self._bounding_box_summary(features['frame2'], out, 'frame2_predicted_box')

        center1 = (features['box'][:, 0:2] + features['box'][:, 2:4]) / 2.
        tf.summary.histogram('center1 check', center1)

        center2 = (labels[:, 0:2] + labels[:, 2:4]) / 2.

        bounding_box_loss = tf.losses.absolute_difference(labels=center2,
                                             predictions=out,
                                             weights=76 * tf.abs(center2 - center1) + 1.)
        tf.summary.scalar('bounding_box_loss', bounding_box_loss)

        self._predicted_offset_summary(center1, center2, out)

        # TRAIN mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
            train_op = optimizer.minimize(
                loss=bounding_box_loss + 0.1 * conv_loss,
                global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=bounding_box_loss + conv_loss, train_op=train_op)

        # evaluation mode
        eval_metrics_ops = self._get_eval_metrics_ops(center2, out)

        return tf.estimator.EstimatorSpec(mode=mode, loss=bounding_box_loss, eval_metric_ops=eval_metrics_ops)

    def _get_eval_metrics_ops(self, labels, predictions):
        eval_metrics_ops = {
            'l1_loss': tf.metrics.mean_absolute_error(labels=labels, predictions=predictions),
            'l2_loss': tf.metrics.mean_squared_error(labels=labels, predictions=predictions),
            'x_mean': tf.metrics.mean(predictions[:, 1]),
            'y_mean': tf.metrics.mean(predictions[:, 0]),
            'x_variance': tf.metrics.mean(tf.square(predictions[:, 1] - tf.reduce_mean(predictions[:, 1]))),
            'y_variance': tf.metrics.mean(tf.square(predictions[:, 0] - tf.reduce_mean(predictions[:, 0]))),
            'x_corvar': tf.metrics.mean(tf.multiply(predictions[:, 1] - tf.reduce_mean(predictions[:, 1]),
                                                    labels[:, 1] - tf.reduce_mean(labels[:, 1]))),
            'y_corvar': tf.metrics.mean(tf.multiply(predictions[:, 0] - tf.reduce_mean(predictions[:, 0]),
                                                    labels[:, 0] - tf.reduce_mean(labels[:, 0]))),
            'x_correlation': tf.contrib.metrics.streaming_pearson_correlation(predictions=predictions[:, 1],
                                                                                      labels=labels[:, 1]),
            'y_correlation': tf.contrib.metrics.streaming_pearson_correlation(predictions=predictions[:, 0],
                                                                                      labels=labels[:, 0]),
        }
        return eval_metrics_ops

    def _dense_layer(self, x, units, name, act=tf.nn.tanh):
        # matches keras weight initializer
        dense = tf.layers.dense(x, units=units,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.initializers.zeros,
                                name=name)
        tf.summary.histogram('before_nonlinearity_%s' % name, dense)
        act = act(dense)
        tf.summary.histogram('after_nonlinearity_%s' % name, act)
        return act

    def _bounding_box_summary(self, images, boxes, name):
        boxes_3d = tf.stack([boxes, boxes, boxes], axis=1)
        annotated_im = tf.image.draw_bounding_boxes(images, boxes_3d)
        tf.summary.image(name, annotated_im, max_outputs=MAX_IMG_OUTPUTS)

    def _predicted_offset_summary(self, center1, center2, pred):
        with tf.variable_scope('offset_summary'):
            true_offset = center2 - center1
            true_offset_mean, true_offset_var = tf.nn.moments(true_offset, axes=0)

            pred_offset = pred - center1
            pred_offset_mean, pred_offset_var = tf.nn.moments(pred_offset, axes=0)

            covar_x = tf.reduce_mean(tf.multiply(true_offset[:, 1] - tf.reduce_mean(true_offset[:, 1]),
                                                 pred_offset[:, 1] - tf.reduce_mean(pred_offset[:, 1])))
            covar_y = tf.reduce_mean(tf.multiply(true_offset[:, 0] - tf.reduce_mean(true_offset[:, 0]),
                                                 pred_offset[:, 0] - tf.reduce_mean(pred_offset[:, 0])))

            tf.summary.histogram('true_offset_x', true_offset[:, 1])
            tf.summary.histogram('true_offset_y', true_offset[:, 0])
            tf.summary.histogram('pred_offset_x', pred_offset[:, 1])
            tf.summary.histogram('pred_offset_y', pred_offset[:, 0])

            tf.summary.scalar('true_offset_var_x', true_offset_var[1])
            tf.summary.scalar('true_offset_var_y', true_offset_var[0])

            tf.summary.scalar('pred_mean_x_offset', pred_offset_mean[1])
            tf.summary.scalar('pred_mean_y_offset', pred_offset_mean[0])
            tf.summary.scalar('pred_var_x_offset', pred_offset_var[1])
            tf.summary.scalar('pred_var_y_offset', pred_offset_var[0])
            tf.summary.scalar('offset_covariance_x', covar_x)
            tf.summary.scalar('offset_covariance_y', covar_y)

            tf.summary.scalar('offset_correlation_x', covar_x / tf.square(true_offset_var[1] * pred_offset_var[1]))
            tf.summary.scalar('offset_correlation_y', covar_y / tf.square(true_offset_var[0] * pred_offset_var[0]))