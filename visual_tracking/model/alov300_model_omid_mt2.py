import pickle

import tensorflow as tf
from keras import  backend as K
import numpy as np

from tuning import SpeedTuning, DirectionTuning
from visual_tracking.model.alov300_model_base import ALOV300ModelBase


FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path, speed_scaler, **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = 64
        self.speed_scaler = speed_scaler
        super(MTTracker, self).__init__(**kwargs)

    def __call__(self, features, labels, mode):

        with tf.name_scope('speed_input'):
            speed_input = tf.identity(features['speed'])

            tf.summary.histogram('speed_input', speed_input)
            tf.summary.image('speed', tf.expand_dims(speed_input, axis=3), max_outputs=self.max_im_outputs)

        with tf.name_scope('direction_input'):
            direction_input = tf.identity(features['direction'])

            tf.summary.histogram('direction_input', direction_input)
            tf.summary.image('direction', tf.expand_dims(direction_input, axis=3), max_outputs=self.max_im_outputs)

        with tf.name_scope('mt_tunning_layers'):
            speed_tun_layer = SpeedTuning(64, self.mt_params, self.speed_scaler, name='speed_tunning')
            direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')

            speed_tun_layer = speed_tun_layer(speed_input)
            direction_tun_layer = direction_tun_layer(direction_input)

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
                            scope_name='conv1',
                            act=tf.nn.elu,
                            batch_norm=True,
                            dropout=0.2,
                            kernel_l2_reg_scale=1e-4)

        with tf.name_scope('attention_mask'):
            masks = tf.expand_dims(features['mask'], 3)
            tf.summary.image('attention_mask', masks, max_outputs=self.max_im_outputs)

            masked = tf.concat([conv, masks], axis=3)

        conv = self._conv2d(masked,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=128,
                            padding='valid',
                            scope_name='conv2',
                            batch_norm=True,
                            act=tf.nn.elu,
                            dropout=0.2,
                            kernel_l2_reg_scale=1e-4)

        conv = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))

        conv = self._conv2d(conv,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=128,
                            padding='valid',
                            scope_name='conv3',
                            batch_norm=True,
                            act=tf.nn.elu,
                            dropout=0.2,
                            kernel_l2_reg_scale=1e-4)

        pool = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))

        tf.summary.image('pool_l2_norm', tf.norm(pool, ord=2, axis=3, keep_dims=True), max_outputs=self.max_im_outputs)
        tf.summary.histogram('pool', pool)

        pool_flatten = tf.layers.flatten(pool)
        pool_flatten = tf.layers.dropout(pool_flatten, 0.1)

        dense = self._dense(pool_flatten,
                            units=4096,
                            name='dense1',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=1e-5)

        dense = self._dense(dense,
                            units=1024,
                            name='dense2',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=1e-5)

        dense = self._dense(dense,
                            units=256,
                            name='dense3',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=1e-5)

        p_delta = self._dense(dense,
                            units=4,
                            name='dense4',
                            act=None,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.)

        pbbox = p_delta + features['box']

        return self._compile(mode=mode,
                             frame1=features['frame1'],
                             frame2=features['frame2'],
                             prev_box=features['box'],
                             labels=labels,
                             pbbox=pbbox,
                             p_delta=p_delta)

    def _flow_decoding(self, conv, direction_input, speed_input):

        with tf.name_scope('conv_supervise'):
            flow_input_x = tf.multiply(speed_input, tf.cos(direction_input))
            flow_input_y = tf.multiply(speed_input, tf.sin(direction_input))

            tf.summary.image('flow_x_input_l1', tf.expand_dims(tf.abs(flow_input_x), axis=3),
                             max_outputs=self.max_im_outputs)
            tf.summary.image('flow_y_input_l1', tf.expand_dims(tf.abs(flow_input_y), axis=3),
                             max_outputs=self.max_im_outputs)
            tf.summary.image('conv3_batch_normed_l2_norm', tf.norm(conv, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

            tf.summary.histogram('flow_input_x', flow_input_x)
            tf.summary.histogram('flow_input_y', flow_input_y)
            tf.summary.histogram('conv3_batch_norm', conv)

            conv_loss = tf.losses.mean_squared_error(labels=tf.stack([flow_input_y, flow_input_x], axis=3),
                                                     predictions=conv)
            tf.summary.scalar('conv_loss', conv_loss)

        return conv_loss, flow_input_x, flow_input_y