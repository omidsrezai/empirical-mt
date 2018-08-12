import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K

from visual_tracking.model.alov300_model_base import ALOV300ModelBase
from visual_tracking.model.area_mst import AreaMST
from visual_tracking.model.area_mt import AreaMT
from visual_tracking.model.layer_tools import dense, conv2d, time_map

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTMSTSeqTracker(ALOV300ModelBase):
    def __init__(self, mt_params_path,
                 mt_attention_gain_path,
                 speed_scalar,
                 **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.mt_attention_gains = np.load(mt_attention_gain_path).astype(np.float32)
        self.speed_scalar = speed_scalar
        super(MTMSTSeqTracker, self).__init__(**kwargs)

    def __call__(self, features, labels, mode):
        with tf.name_scope('speed_inputs'):
            speed_inputs = tf.identity(features['speed'])
            avg_speed = tf.reduce_mean(speed_inputs, axis=1)

            tf.summary.histogram('speed_input', speed_inputs)
            tf.summary.image('avg_speed',
                             tf.expand_dims(avg_speed, axis=3),
                             max_outputs=self.max_im_outputs)

        with tf.name_scope('direction_inputs'):
            direction_input = tf.identity(features['direction'])
            avg_direction = tf.reduce_mean(direction_input, axis=1)

            tf.summary.histogram('direction_input', direction_input)
            tf.summary.image('avg_direction',
                             tf.expand_dims(avg_direction, axis=3),
                             max_outputs=self.max_im_outputs)

        with tf.name_scope('speed_input_tents'):
            speed_input_tents = tf.identity(features['speed_tents'])
            tf.summary.histogram('speed_input_tents', speed_input_tents)
            tf.summary.image('speed_input_tents_l1_time_0',
                             tf.norm(speed_input_tents[:, 0], axis=3, ord=1, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        with tf.name_scope('saliency'):
            saliencymaps = tf.identity(features['saliency'])

            tf.summary.histogram('saliency', saliencymaps)
            tf.summary.image('saliency_maps_t5',
                             tf.expand_dims(saliencymaps[:, -1, :, :], axis=3),
                             max_outputs=self.max_im_outputs)

        with tf.variable_scope('mt_over_time'):
            area_mt = AreaMT(max_im_outputs=4,
                             n_chann=64,
                             empirical_excitatory_params=self.mt_params,
                             speed_scalar=self.speed_scalar,
                             chann_sel_dp=0.,
                             activity_dp=0.,
                             attention_gains=self.mt_attention_gains,
                             conv_chann=64,
                             l2_reg_scale=0.005)

            mt_activity = time_map((speed_inputs, speed_input_tents, direction_input),
                                   area_mt,
                                   name='area_mt')

            tf.summary.histogram('mt_activity', mt_activity)
            tf.summary.image('mt_activity_time_avg',
                             tf.norm(tf.reduce_mean(mt_activity, axis=1), ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        with tf.variable_scope('mst_over_time'):
            area_mst = AreaMST(n_chann=64,
                               max_im_outputs=4,
                               dropout=0.,
                               l2_reg_scale=0.005)
            mst_activity = time_map(mt_activity, area_mst, 'area_mst')

            tf.summary.histogram('mst_activity', mst_activity)

        with tf.variable_scope('pool_over_time'):
            time_pooled = tf.reduce_mean(mst_activity, axis=1)
            time_pooled = tf.layers.batch_normalization(time_pooled)

            tf.summary.histogram('mst_activity_time_avg', time_pooled)
            tf.summary.image('mst_activity_time_avg',
                             tf.norm(time_pooled, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)


        conv1 = conv2d(time_pooled,
                       kernel_size=(3, 3),
                       filters=128,
                       max_pool=None,
                       strides=(1, 1),
                       batch_norm=True,
                       dropout=0.,
                       act=tf.nn.elu,
                       name='3x3conv128',
                       kernel_l2_reg_scale=0.01)

        dense1 = dense(tf.layers.flatten(conv1),
                       units=256,
                       name='dense256',
                       act=tf.nn.elu,
                       batch_norm=True,
                       kernel_l2_reg_scale=0.01)

        dense2 = dense(dense1,
                       units=64,
                       name='dense64',
                       act=tf.nn.elu,
                       batch_norm=True,
                       kernel_l2_reg_scale=0.01)

        pbbox = dense(dense2,
                      units=4,
                      name='predictions',
                      act=lambda x: (3 * tf.nn.sigmoid(x)) - 1,
                      batch_norm=False,
                      kernel_l2_reg_scale=0.)

        return self._compile(mode=mode,
                             frame1=features['frames'][:, 0],
                             frame2=features['frames'][:, -1],
                             prev_box=features['bbox'],
                             ground_truth_box=labels,
                             y_hat=pbbox,
                             pred_box=pbbox,
                             y=labels)
