import pickle

import tensorflow as tf
from keras import backend as K

from tuning import SpeedTuning, DirectionTuning
from visual_tracking.model.alov300_model_base import ALOV300ModelBase

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class SeqMTTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path,
                 speed_scaler,
                 n_chann=64,
                 **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = n_chann
        self.speed_scaler = speed_scaler
        super(SeqMTTracker, self).__init__(**kwargs)

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

        with tf.name_scope('mt_tunning_layers'):
            speed_tun_layer = SpeedTuning(64, self.mt_params, self.speed_scaler, name='speed_tunning')
            direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')

            speed_tun_act = self._time_distributed(speed_inputs, speed_tun_layer, name='speed_tun')

            direction_tun_act = self._time_distributed(direction_input,
                                                       direction_tun_layer,
                                                       name='direction_tun')
            mt_activity = tf.multiply(speed_tun_act,
                                      direction_tun_act,
                                      name='mt_act_tensor')

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

        conv = mt_activity
        for id, n_filters in zip([1, 2, 3], [64, 64, 64]):
            conv = self._time_dist_conv_batch_norm_pool_dp(conv, id,
                                                           n_filters=n_filters,
                                                           dp=0.1,
                                                           batch_norm=True,
                                                           pool=False,
                                                           padding='same')

        with tf.variable_scope('weighted_average') as scope:
            flattened = self._time_distributed(conv, tf.layers.Flatten(), 'flatten')
            w_average = tf.layers.conv1d(flattened,
                                          filters=1,
                                          kernel_size=1,
                                          strides=1,
                                          data_format='channels_first',
                                          name='weighted_avg',
                                          kernel_constraint=tf.keras.constraints.non_neg(),
                                          use_bias=False)
            w_average = tf.reshape(w_average, [-1, 76, 76, 64])

            tf.summary.image('weighted_average',
                             tf.norm(w_average, ord=2, axis=3, keep_dims=True),
                             max_outputs=self.max_im_outputs)

            # plot the weight for each temporal dimension
            scope.reuse_variables()
            weights = tf.squeeze(tf.get_variable('weighted_avg/kernel'))
            for d in range(weights.shape[0]):
                tf.summary.scalar('weighted_dim_%s' % d, weights[d])


        with tf.name_scope('attention_mask'):
            masks = tf.expand_dims(features['mask'], 3)
            tf.summary.image('attention_mask', masks, max_outputs=self.max_im_outputs)

            masked = tf.concat([w_average, masks], axis=3)

        conv = masked
        for id, n_filters in zip([4, 5, 6], [64, 64, 128]):
            conv = self._conv2d(conv,
                                kernel_size=(3, 3),
                                filters=n_filters,
                                strides=(1, 1),
                                name='conv%s' % id,
                                act=tf.nn.elu,
                                batch_norm=True,
                                padding='valid',
                                dropout=0.1,
                                pool=True)

        dense = self._dense(tf.layers.flatten(conv),
                            units=4096,
                            name='dense1',
                            act=tf.nn.elu,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.05)

        dense = self._dense(dense,
                            units=512,
                            name='dense2',
                            act=tf.nn.elu,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.)

        dense = self._dense(dense,
                            units=64,
                            name='dense3',
                            act=tf.nn.elu,
                            batch_norm=True,
                            kernel_l2_reg_scale=0.)

        p_delta = self._dense(dense,
                              units=4,
                              name='predictions',
                              act=None,
                              batch_norm=False,
                              kernel_l2_reg_scale=0.)

        pbbox = p_delta + features['bbox']

        return self._compile(mode=mode,
                             frame1=features['frames'][:, 0],
                             frame2=features['frames'][:, -1],
                             prev_box=features['bbox'],
                             ground_truth_box=labels,
                             y_hat=p_delta,
                             pred_box=pbbox,
                             y=labels - features['bbox'] if labels is not None else None)

    def _time_distributed(self, xs, f, name):
        with tf.variable_scope('time_dist_%s' % name):
            ys = tf.stack([f(xs[:, i]) for i in range(xs.shape[1])], axis=1)
        return ys

    def _time_dist_conv_batch_norm_pool_dp(self, x, id,
                                           n_filters=64,
                                           dp=0.,
                                           batch_norm=True,
                                           pool=True,
                                           padding='same'):
        conv2d = tf.layers.Conv2D(kernel_size=(3, 3),
                                    strides=(1, 1),
                                    filters=n_filters,
                                    padding=padding,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    bias_initializer=tf.initializers.zeros,
                                    activation=tf.nn.elu,
                                    name='conv%s' % id)
        conv = self._time_distributed(x, conv2d, name='conv%s' % id)

        if batch_norm:
            conv = self._time_distributed(conv,
                                          tf.layers.BatchNormalization(), name='conv%s_batch_norm' % id)

        if pool:
            conv = self._time_distributed(conv,
                                          tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name='pool%s' % id)

        tf.summary.image('avg_conv%s_l2_norm' % id,
                         tf.norm(tf.reduce_mean(conv, axis=1), ord=2, axis=3, keep_dims=True),
                         max_outputs=self.max_im_outputs)

        if dp > 0:
            conv = self._time_distributed(conv,
                                          tf.layers.Dropout(dp),
                                          name='pool%s_dropout' % id)

        return conv
