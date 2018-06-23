import pickle

import tensorflow as tf
from keras import  backend as K
import numpy as np

# link keras session and tensorflow session
import MT
from tuning import SpeedTuning, DirectionTuning

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTFeatureDecoder(object):

    def __init__(self, mt_params_path):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = 64

        self._tanh_dense_layer_count = 0

    def __call__(self, features, labels, mode):
        self._tanh_dense_layer_count = 0

        speed_input = features['speed']
        direction_input = features['direction']
        tf.summary.histogram('speed_input', speed_input)
        tf.summary.histogram('direction_input', direction_input)

        speed_image = tf.reshape(speed_input, [-1, 76, 76, 1])
        tf.summary.image('speed', speed_image, max_outputs=10)
        direction_image = tf.reshape(direction_input, [-1, 76, 76, 1])
        tf.summary.image('direction', direction_image, max_outputs=10)


        speed_tun_layer = SpeedTuning(64, self.mt_params, 0.5, name='speed_tunning')
        direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')

        speed_tun_layer = speed_tun_layer([speed_input, tf.zeros_like(speed_input)])
        direction_tun_layer = direction_tun_layer(direction_input)
        mt_act = tf.multiply(speed_tun_layer, direction_tun_layer, name='mt_act_tensor')

        # visualize speeding tuning and direction tuning
        tf.summary.histogram("speed_tuning", speed_tun_layer)
        speed_tun_inf_norm = tf.norm(speed_tun_layer, ord=np.inf, axis=3, keep_dims=True)
        speed_tun_l2 = tf.norm(speed_tun_layer, ord=2, axis=3, keep_dims=True)
        tf.summary.image('speed_tun_inf_norm', speed_tun_inf_norm, max_outputs=10)
        tf.summary.image('speed_tun_l2_norm', speed_tun_l2, max_outputs=10)

        tf.summary.histogram("direction_tuning", direction_tun_layer)
        tf.summary.histogram('activations', mt_act)
        mt_act_l2 = tf.norm(mt_act, ord=2, axis=3, keep_dims=True)
        mt_act_inf = tf.norm(mt_act, ord=np.inf, axis=3, keep_dims=True)
        tf.summary.image('mt_act_2_norm', mt_act_l2, max_outputs=10)
        tf.summary.image('mt_act_inf_norm', mt_act_inf, max_outputs=10)

        conv = tf.layers.conv2d(mt_act, kernel_size=(3, 3), strides=(1, 1), filters=64, padding='same', activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, kernel_size=(3, 3), strides=(1, 1), filters=2, padding='same')

        flow = tf.reshape(conv, [-1, 76, 76, 2])

        p_speed = tf.expand_dims(flow[:, :, :, 0], axis=3)
        p_speed = tf.nn.relu(p_speed)

        p_direction = tf.expand_dims(flow[:, :, :, 1], axis=3)
        p_direction = np.pi / 2 * tf.nn.tanh(p_direction)

        flow = tf.squeeze(tf.stack([p_speed, p_direction], axis=3))

        tf.summary.image('predicted_speed', p_speed, max_outputs=10)
        tf.summary.image('predicted_direction', p_direction, max_outputs=10)

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            # rescale it to coordinates
            predictions = {
                'input_speed': speed_input,
                'input_direction': direction_input,
                'pred_speed': p_speed,
                'pred_direction': p_direction
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.mean_squared_error(labels=tf.stack([speed_input, direction_input], axis=3), predictions=flow)

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

