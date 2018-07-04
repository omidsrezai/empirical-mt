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

        #features['direction'].set_shape([64, 5, 76, 76])
        #features['speed'].set_shape([64, 5, 76, 76])

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

            speed_tun_act = self._time_distributed(speed_inputs, speed_tun_layer, name='speed_tun')

            direction_tun_act = self._time_distributed(direction_input, direction_tun_layer, name='direction_tun')
            mt_activity = tf.multiply(speed_tun_act, direction_tun_act, name='mt_act_tensor')

            # visualize speeding tuning and direction tuning
            tf.summary.histogram("speed_tuning", speed_tun_act)
            tf.summary.histogram("direction_tuning", direction_tun_act)
            tf.summary.histogram('activations', mt_activity)

            tf.summary.image('avg_speed_tun_l2_norm',
                             tf.norm(tf.reduce_mean(speed_tun_act, axis=1), ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)
            tf.summary.image('avg_direction_tun_l2_norm',
                             tf.norm(tf.reduce_mean(direction_tun_act, axis=1), ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)
            tf.summary.image('avg_mt_act_2_norm',
                             tf.norm(tf.reduce_mean(mt_activity, axis=1), ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        mt_activity = self._time_distributed(mt_activity,
                                             tf.layers.BatchNormalization(name='mt_act_batch_norm'),
                                             name='mt_act_batch_norm')
        tf.summary.image('avg_batch_normed_mt_act',
                         tf.norm(tf.reduce_mean(mt_activity, axis=1), ord=2, axis=3, keep_dims=True))

        conv2d_1 = tf.layers.Conv2D(kernel_size=(3, 3),
                                        strides=(1, 1),
                                        filters=64,
                                        padding='same',
                                        activation=tf.nn.elu,
                                        name='conv1')

        conv = self._time_distributed(mt_activity, conv2d_1, name='conv1')

        with tf.name_scope('attention_mask'):
            masks = tf.expand_dims(features['mask'], 3)
            tf.summary.image('attention_mask', masks, max_outputs=self.max_im_outputs)

            masked = self._time_distributed(conv, lambda x: tf.concat([x, masks], axis=3), name='add_mask')

        conv2d_2 = tf.layers.Conv2D(kernel_size=(3, 3),
                                        strides=(1, 1),
                                        filters=64,
                                        padding='valid',
                                        activation=tf.nn.elu,
                                        name='conv2')

        conv = self._time_distributed(masked, conv2d_2, name='conv2')
        pool = self._time_distributed(conv, tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name='pool2')

        conv2d_3 = tf.layers.Conv2D(kernel_size=(3, 3),
                                    strides=(1, 1),
                                    filters=64,
                                    padding='valid',
                                    activation=tf.nn.elu,
                                    name='conv2')

        conv = self._time_distributed(pool, conv2d_3, name='conv3')
        pool = self._time_distributed(conv, tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name='pool3')

        print(pool.shape)

        tf.summary.image('last_pool_avg_l2_norm',
                         tf.norm(tf.reduce_mean(pool, axis=1), ord=2, axis=3, keep_dims=True),
                         max_outputs=self.max_im_outputs)
        tf.summary.histogram('pool', pool)

        pool_flatten = self._time_distributed(pool, tf.layers.Flatten(), name='flatten')

        print(pool_flatten.shape)

        weight_avg = tf.layers.conv1d(pool_flatten,
                                      filters=1,
                                      kernel_size=1,
                                      strides=1,
                                      data_format='channels_first',
                                      name='weighted_avg')

        weight_avg = tf.squeeze(weight_avg, axis=1)

        print(weight_avg.shape)
        # pool_flatten = tf.layers.dropout(pool_flatten, 0.1)

        dense = self._dense(weight_avg,
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

        p_delta = self._dense(dense,
                             units=4,
                             name='dense4',
                             act=None,
                             batch_norm=True,
                             kernel_l2_reg_scale=0.)

        pbbox = p_delta + features['bbox']

        return self._compile(mode=mode,
                             frame1=features['frames'][:, 0],
                             frame2=features['frames'][:, -1],
                             prev_box=features['bbox'],
                             labels=labels,
                             p_delta=p_delta,
                             pbbox=pbbox)

    def _time_distributed(self, xs, f, name):
        with tf.variable_scope('time_dist_%s' % name):
            ys = tf.stack([f(xs[:, i]) for i in range(xs.shape[1])], axis=1)

        return ys
