import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K

from visual_tracking.model.alov300_model_base import ALOV300ModelBase
from visual_tracking.model.area_mst import AreaMST
from visual_tracking.model.area_mt import AreaMT
from visual_tracking.model.layer_tools import dense, conv2d

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

                    y_left = (x - _left) / (_center - _left)
                    y_right = (_right - x) / (_right - _center)

                    y = tf.where((x >= _left) & (x <= _center), y_left, tf.zeros_like(x)) \
                        + tf.where((x >= _center) & (x <= _right), y_right, tf.zeros_like(x))

                    tent_basis.append(y)

                return tf.stack(tent_basis, axis=3)

            speed_input_tents = self._time_map(speed_inputs * self.speed_scalar,
                                               _project_tent_basis,
                                               'project_tents')

            tf.summary.histogram('speed_input_tents', speed_input_tents)
            tf.summary.image('speed_input_tents_l1_time_0',
                             tf.norm(speed_input_tents[:, 0], axis=3, ord=1, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        with tf.variable_scope('mt_over_time'):
            area_mt = AreaMT(max_im_outputs=4,
                             n_chann=32,
                             empirical_excitatory_params=self.mt_params,
                             speed_scalar=self.speed_scalar,
                             chann_sel_dp=0.,
                             activity_dp=0.,
                             l2_reg_scale=0.01)
            mt_activity = self._time_map((speed_inputs, speed_input_tents, direction_input),
                                         area_mt,
                                         name='area_mt')

            tf.summary.histogram('mt_activity', mt_activity)
            tf.summary.image('mt_activity_time_avg',
                             tf.norm(tf.reduce_mean(mt_activity, axis=1), ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        with tf.variable_scope('mst_over_time'):
            area_mst = AreaMST(n_chann=64, max_im_outputs=4, dropout=0., l2_reg_scale=0.01)
            mst_activity = self._time_map(mt_activity, area_mst, 'area_mst')

            tf.summary.histogram('mst_activity', mst_activity)

        with tf.variable_scope('pool_over_time'):
            time_pooled = tf.reduce_mean(mst_activity, axis=1)
            time_pooled = tf.layers.batch_normalization(time_pooled)

            tf.summary.histogram('mst_activity_time_avg', time_pooled)
            tf.summary.image('mst_activity_time_avg',
                             tf.norm(time_pooled, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

        with tf.variable_scope('add_prev_bbox'):
            prev_bbox = tf.expand_dims(features['mask'], axis=3)
            tf.summary.image('mask', prev_bbox, max_outputs=self.max_im_outputs)
            prev_bbox = tf.layers.average_pooling2d(prev_bbox, pool_size=(19, 19), strides=(19, 19))

            masked = tf.concat([time_pooled, prev_bbox], axis=3)

            time_pooled_masked = conv2d(masked,
                                        kernel_size=(3, 3),
                                        filters=64,
                                        max_pool=None,
                                        strides=(1, 1),
                                        batch_norm=True,
                                        dropout=0.,
                                        act=tf.nn.elu,
                                        name='conv_with_mask',
                                        kernel_l2_reg_scale=0.01)

        dense1 = dense(tf.layers.flatten(time_pooled_masked),
                       units=256,
                       name='dense1',
                       act=tf.nn.elu,
                       batch_norm=True,
                       kernel_l2_reg_scale=0.01)

        dense2 = dense(dense1,
                       units=64,
                       name='dense2',
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

    def _time_map(self, x_timesteps, f, name):
        with tf.variable_scope('time_map_%s' % name):
            y_timesteps = []
            ts = x_timesteps[0].shape[1] if isinstance(x_timesteps, tuple) else x_timesteps.shape[1]

            for i in range(0, ts):
                y_timestep = f(*[x[:, i] for x in x_timesteps]) \
                    if isinstance(x_timesteps, tuple) else f(x_timesteps[:, i])
                y_timesteps.append(y_timestep)

            y_timesteps = tf.stack(y_timesteps, axis=1)

        return y_timesteps
