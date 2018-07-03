import pickle

import tensorflow as tf
from keras import  backend as K
import numpy as np

from tuning import SpeedTuning, DirectionTuning
from visual_tracking.model.alov300_model_base import ALOV300ModelBase

LEARNING_RATE = 0.00001 # was 0.0001
FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class SeqMTTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path, speed_scaler, **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = 64
        self.speed_scaler = speed_scaler
        super(SeqMTTracker, self).__init__(**kwargs)

    def __call__(self, features, labels, mode):

        with tf.name_scope('speed_inputs'):
            speed_inputs = tf.identity(features['speed'])
            avg_speed = tf.reduce_mean(speed_inputs, axis=1)

            tf.summary.histogram('speed_input', speed_inputs)
            tf.summary.image('speed', tf.expand_dims(avg_speed, axis=3), max_outputs=self.max_im_outputs)

        with tf.name_scope('direction_inputs'):
            direction_input = tf.identity(features['direction'])
            avg_direction = tf.reduce_mean(direction_input, axis=1)

            tf.summary.histogram('direction_input', direction_input)
            tf.summary.image('direction', tf.expand_dims(avg_direction, axis=3), max_outputs=self.max_im_outputs)

        with tf.name_scope('mt_tunning_layers'):
            speed_tun_layer = SpeedTuning(64, self.mt_params, self.speed_scaler, name='speed_tunning')
            direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')

            speed_tun_layer = speed_tun_layer([avg_speed, tf.zeros_like(avg_speed)])
            direction_tun_layer = direction_tun_layer(avg_direction)

            mt_activity = tf.multiply(speed_tun_layer, direction_tun_layer, name='mt_act_tensor')

            # visualize speeding tuning and direction tuning
            tf.summary.histogram("speed_tuning", speed_tun_layer)
            tf.summary.histogram("direction_tuning", direction_tun_layer)
            tf.summary.histogram('activations', mt_activity)

            tf.summary.image('speed_tun_l2_norm',
                             tf.norm(speed_tun_layer, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)
            tf.summary.image('direction_tun_l2_norm',
                             tf.norm(direction_tun_layer, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)
            tf.summary.image('mt_act_2_norm',
                             tf.norm(mt_activity, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        mt_activity = tf.layers.batch_normalization(mt_activity, name='mt_act_batch_norm')
        tf.summary.image('batch_normed_mt_act', tf.norm(mt_activity, ord=2, axis=3, keep_dims=True))

        conv = self._conv2d(mt_activity,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=64,
                            padding='same',
                            name='conv1',
                            act=tf.nn.elu,
                            batch_norm=True,
                            dropout=0.2,
                            kernel_l2_reg_scale=0.0)

        with tf.name_scope('attention_mask'):
            masks = tf.expand_dims(features['mask'], 3)
            tf.summary.image('attention_mask', masks, max_outputs=self.max_im_outputs)

            masked = tf.concat([conv, masks], axis=3)

        conv = self._conv2d(masked,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=64,
                            padding='valid',
                            name='conv2',
                            batch_norm=True,
                            act=tf.nn.elu,
                            dropout=0.2,
                            kernel_l2_reg_scale=0.0)

        conv = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))

        conv = self._conv2d(conv,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=64,
                            padding='valid',
                            name='conv3',
                            batch_norm=True,
                            act=tf.nn.elu,
                            dropout=0.2,
                            kernel_l2_reg_scale=0.0)

        pool = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))

        tf.summary.image('pool_l2_norm', tf.norm(pool, ord=2, axis=3, keep_dims=True), max_outputs=self.max_im_outputs)
        tf.summary.histogram('pool', pool)

        pool_flatten = tf.layers.flatten(pool)
        # pool_flatten = tf.layers.dropout(pool_flatten, 0.1)

        dense = self._dense(pool_flatten,
                            units=1024,
                            name='dense2',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.)

        dense = self._dense(dense,
                            units=256,
                            name='dense3',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.)

        pbbox = self._dense(dense,
                            units=4,
                            name='dense4',
                            act=tf.nn.sigmoid,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.)

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pbbox)

        # TRAIN mode
        with tf.name_scope('frame_pairs'):
            frame1 = features['frames'][:, 0, :, :, :]
            frame2 = features['frames'][:, -1, :, :, :]

            with tf.name_scope('frame1'):
                self._summary_images_with_bbox(frame1, features['bbox'], name='frame1_box')

            with tf.name_scope('frame2'):
                self._summary_images_with_bbox(frame2, labels, name='frame2_box')
                self._summary_images_with_bbox(frame2, pbbox, name='frame2_predicted_box')

        self._predicted_delta_summary(prev_bbox=features['bbox'], p_bbox=pbbox, t_bbox=labels)

        bbox_loss =tf.losses.mean_squared_error(labels=labels, predictions=pbbox)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.minimize(
                loss=bbox_loss,
                global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, train_op=train_op)

        # EVAL mode
        eval_metrics_ops = self._get_eval_metrics_ops(predictions=pbbox - features['bbox'],
                                                      labels=labels - features['bbox'])

        return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, eval_metric_ops=eval_metrics_ops)
