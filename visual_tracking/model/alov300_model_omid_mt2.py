import pickle

import tensorflow as tf
from keras import backend as K

from tuning import SpeedTuning, DirectionTuning
from visual_tracking.model.alov300_model_base import ALOV300ModelBase

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path, speed_scaler, n_chann=64, **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = n_chann
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
            speed_tun_layer = SpeedTuning(self.n_chann, self.mt_params, self.speed_scaler, name='speed_tunning')
            direction_tun_layer = DirectionTuning(self.n_chann, self.mt_params, name='direction_tunning')

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

        conv = mt_activity
        for id, n_filters in zip([1, 2], [64, 64]):
            conv = self._conv2d(conv,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                filters=n_filters,
                                padding='same',
                                name='conv%s' % id,
                                act=tf.nn.elu,
                                batch_norm=True,
                                dropout=0.1,
                                kernel_l2_reg_scale=0.,
                                pool=False)

        with tf.name_scope('attention_mask'):
            masks = tf.expand_dims(features['mask'], 3)
            #tf.summary.image('attention_mask', masks, max_outputs=self.max_im_outputs)

            masked = tf.concat([conv, masks], axis=3)

        conv = masked
        for id, n_filters in zip([3, 4, 5, 6], [128, 128, 128, 128]):
            conv = self._conv2d(conv,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                filters=n_filters,
                                padding='valid',
                                name='conv%s' % id,
                                batch_norm=True,
                                act=tf.nn.elu,
                                dropout=0.1,
                                kernel_l2_reg_scale=1e-5,
                                pool=True)

        pool_flatten = tf.layers.flatten(conv)
        #pool_flatten = tf.layers.dropout(pool_flatten, 0.1)

        dense = self._dense(pool_flatten,
                            units=256,
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

        p_delta = self._dense(dense,
                            units=4,
                            name='predictions',
                            act=None,
                            batch_norm=False,
                            kernel_l2_reg_scale=0.)

        pbbox = p_delta + features['box']

        return self._compile(mode=mode,
                             frame1=features['frame1'],
                             frame2=features['frame2'],
                             prev_box=features['box'],
                             ground_truth_box=labels,
                             pred_box=pbbox,
                             y_hat=p_delta,
                             y=labels - features['box'] if labels is not None else None)

