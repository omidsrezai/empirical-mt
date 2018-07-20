import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K

from visual_tracking.model.alov300_model_base import ALOV300ModelBase
from visual_tracking.model.area_mt import AreaMT

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTMSTPairwiseTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path, speed_scaler, n_chann=64, **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = n_chann
        self.speed_scaler = speed_scaler
        super(MTMSTPairwiseTracker, self).__init__(**kwargs)

    def __call__(self, features, labels, mode):

        with tf.name_scope('speed_input'):
            speed_input = tf.identity(features['speed'])

            tf.summary.histogram('speed_input', speed_input)
            tf.summary.image('speed',
                             tf.expand_dims(speed_input, axis=3),
                             max_outputs=self.max_im_outputs)

        # decompose speed input into 10 tent basis functions
        with tf.name_scope('speed_input_tent_basis'):
            speed_input_tents = tf.identity(features['speed_tents'])

            tf.summary.histogram('speed_input_tent_basis', speed_input_tents)
            tf.summary.image('speed_input_tent_basis',
                             tf.norm(speed_input_tents, axis=3, ord=1, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        with tf.name_scope('direction_input'):
            direction_input = tf.identity(features['direction'])

            tf.summary.histogram('direction_input', direction_input)
            tf.summary.image('direction',
                             tf.expand_dims(direction_input, axis=3),
                             max_outputs=self.max_im_outputs)

        area_mt_model = AreaMT(n_chann=self.n_chann,
                               empirical_excitatory_params=self.mt_params,
                               max_im_outputs=self.max_im_outputs,
                               speed_scalar=self.speed_scaler)
        mt_activity = area_mt_model(speed_input=speed_input,
                                    direction_input=direction_input,
                                    speed_input_tents=speed_input_tents)

        conv = tf.layers.batch_normalization(mt_activity)

        with tf.variable_scope('area_mst'):
            for id, n_filters in zip([1, 2], [64, 64]):
                conv = self._conv2d(conv,
                                    kernel_size=(9, 9),
                                    strides=(1, 1),
                                    filters=n_filters,
                                    padding='same',
                                    name='mst_layer%s' % id,
                                    batch_norm=True,
                                    act=tf.nn.relu,
                                    dropout=0.,
                                    kernel_l2_reg_scale=0.,
                                    pool=False)

            mst_activity = tf.layers.max_pooling2d(conv,
                                                   pool_size=(3, 3),
                                                   strides=(3, 3))

        pool_flatten = tf.layers.flatten(mst_activity)
        pool_flatten = tf.layers.dropout(pool_flatten, 0.1)

        # pool_flatten = tf.concat([pool_flatten, features['box']], axis=1)

        dense = self._dense(pool_flatten,
                            units=512,
                            name='dense1',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=5e-6)

        dense = self._dense(dense,
                            units=64,
                            name='dense2',
                            act=tf.nn.tanh,
                            batch_norm=True,
                            kernel_l2_reg_scale=5e-6)

        pbbox = self._dense(dense,
                            units=4,
                            name='predictions',
                            act=None,
                            batch_norm=False,
                            kernel_l2_reg_scale=0.)

        return self._compile(mode=mode,
                             frame1=features['frame1'],
                             frame2=features['frame2'],
                             prev_box=features['box'],
                             ground_truth_box=labels,
                             pred_box=pbbox,
                             y_hat=pbbox,
                             y=labels)