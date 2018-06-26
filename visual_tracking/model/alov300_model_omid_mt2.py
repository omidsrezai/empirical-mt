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
                                padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.initializers.zeros)
        conv = tf.layers.conv2d(conv, kernel_size=(3, 3),
                                strides=(1, 1),
                                filters=2,
                                padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.initializers.zeros)

        pool = tf.layers.average_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))
        tf.summary.image('pool_l2_norm', tf.norm(pool, ord=2, axis=3, keep_dims=True), max_outputs=MAX_IMG_OUTPUTS)

        dense = self._dense_layer(tf.layers.flatten(pool), units=2048, name='dense1')
        dense = self._dense_layer(dense, units=256, name='dense2')
        out = self._dense_layer(dense, units=2, name='dense3', act=tf.nn.sigmoid)

        with tf.variable_scope('previous_frame'):
            self._bounding_box_summary(features['frame1'], features['box'], name='frame1_box')

        with tf.variable_scope('frame'):
            self._bounding_box_summary(features['frame2'], labels, name='frame2_box')

        # with tf.variable_scope('prediction'):
        #    self._bounding_box_summary(features['frame2'], out, 'frame2_predicted_box')

        center1 = (features['box'][:, 0:2] + features['box'][:, 2:4]) / 2.
        center2 = (labels[:, 0:2] + labels[:, 2:4]) / 2.

        loss = tf.losses.absolute_difference(labels=center2, predictions=out)#, weights=tf.abs(offsets + 1.))

        self._predicted_offset_summary(center1, center2, out)

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

    def _dense_layer(self, x, units, name, act=tf.nn.tanh):
        # matches keras weight initializer
        dense = tf.layers.dense(x, units=units,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.initializers.zeros)
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
            true_offset_mean = tf.reduce_mean(true_offset, axis=0)
            pred_offset = pred - center1
            pred_offset_mean = tf.reduce_mean(pred_offset, axis=0)
            pred_offset_var = tf.reduce_mean(tf.square(pred_offset - pred_offset_mean), axis=0)
            covar = tf.reduce_mean((pred_offset - pred_offset_mean) * (true_offset - true_offset_mean), axis=0)

            tf.summary.histogram('true_offset_x', true_offset[:, 1])
            tf.summary.histogram('true_offset_y', true_offset[:, 0])
            tf.summary.histogram('pred_offset_x', pred_offset[:, 1])
            tf.summary.histogram('pred_offset_y', pred_offset[:, 0])

            tf.summary.scalar('pred_mean_x_offset', pred_offset_mean[1])
            tf.summary.scalar('pred_mean_y_offset', pred_offset_mean[0])
            tf.summary.scalar('pred_var_x_offset', pred_offset_var[1])
            tf.summary.scalar('pred_var_y_offset', pred_offset_var[0])
            tf.summary.scalar('offset_covariance_x', covar[1])
            tf.summary.scalar('offset_covariance_y', covar[0])