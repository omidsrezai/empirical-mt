import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K

from visual_tracking.model.alov300_model_base import ALOV300ModelBase
from visual_tracking.model.area_mst import AreaMST
from visual_tracking.model.area_mt import AreaMT
from visual_tracking.model.neural_net_blocks import dense

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTMSTSeqTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path,
                 speed_scalar,
                 n_chann=64,
                 **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = n_chann
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

        # TODO do this in input pipline
        with tf.name_scope('speed_input_tents'):
            tent_centers = np.exp(np.arange(0, 5, .45))

            def _project_tent_basis(x):
                tent_basis = []

                for i in range(0, len(tent_centers) - 2):
                    _left = tent_centers[i]
                    _center = tent_centers[i + 1]
                    _right = tent_centers[i + 2]

                    x = x * self.speed_scalar
                    y_left = (x - _left) / (_center - _left)
                    y_right = (_right - x) / (_right - _center)

                    y = tf.where((x >= _left) & (x <= _center), y_left, tf.zeros_like(x)) \
                        + tf.where((x >= _center) & (x <= _right), y_right, tf.zeros_like(x))

                    tent_basis.append(y)

                return tf.stack(tent_basis, axis=3)

            speed_input_tents = self._time_map(speed_inputs, _project_tent_basis, 'project_tents')

        with tf.variable_scope('mt_over_time'):
            area_mt = AreaMT(max_im_outputs=4,
                             n_chann=self.n_chann,
                             empirical_excitatory_params=self.mt_params,
                             speed_scalar=self.speed_scalar)
            mt_activity = self._time_map((speed_inputs, speed_input_tents, direction_input),
                                         area_mt,
                                         name='area_mt')
            mt_activity = self._time_map(mt_activity, tf.layers.BatchNormalization(), 'mt_act_batch_norm')

            tf.summary.image('mt_activity_time_avg',
                             tf.norm(tf.reduce_mean(mt_activity, axis=1), ord=2, axis=3, keep_dims=True))

        with tf.variable_scope('mst_over_time'):
            area_mst = AreaMST(n_chann=64, max_im_outputs=4)
            mst_activity = self._time_map(mt_activity, area_mst, 'area_mst')
            mst_activity = self._time_map(mst_activity, tf.layers.BatchNormalization(), 'mst_act_batch_norm')

        with tf.variable_scope('avg_over_time'):
            mst_average = tf.reduce_mean(mst_activity, axis=1)
            mst_average = tf.layers.dropout(mst_average, 0.05)

            tf.summary.image('mst_activity_time_avg',
                             tf.norm(mst_average, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        dense_act = dense(tf.layers.flatten(mst_average),
                          units=256,
                          name='dense2',
                          act=tf.nn.elu,
                          batch_norm=True,
                          kernel_l2_reg_scale=0.01)

        dense_act = dense(dense_act,
                          units=64,
                          name='dense3',
                          act=tf.nn.elu,
                          batch_norm=True,
                          kernel_l2_reg_scale=0.01)

        pbbox = dense(dense_act,
                      units=4,
                      name='predictions',
                      act=tf.nn.sigmoid,
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

    def _time_map(self, x_timesteps, f, name):
        with tf.variable_scope('time_map_%s' % name):
            y_timesteps = []
            ts = x_timesteps[0].shape[1] if isinstance(x_timesteps, tuple) else x_timesteps.shape[1]

            for i in range(0, ts):
                y_timestep = f(*[x[:, i] for x in x_timesteps]) \
                    if isinstance(x_timesteps, tuple) else f(x_timesteps[:, i])
                y_timesteps.append(y_timestep)

            return tf.stack(y_timesteps, axis=1)
