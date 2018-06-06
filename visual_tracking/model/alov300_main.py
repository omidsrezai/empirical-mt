import tensorflow as tf
from keras import  backend as K
import numpy as np

# link keras session and tensorflow session
import MT
from tuning import SpeedTuning, DirectionTuning

tf_sess = tf.Session()
K.set_session(tf_sess)


# TODO pass theses as parameters
np.random.seed(1330)
n_channel = 2
n_seq = 1
n_tent = 10
w_ = 76#100
params = MT.sample_tuning_params(n_channel, pref_log_speed_range=(0,4))


def nn_model_fn(features, labels, mode):
    # input layers
    with tf.variable_scope("mt_inputs"):
        speed_input_layer = tf.reshape(features['speed'], [-1, 78, 78], name='speed_input_tensor')
        direction_input_layer = tf.reshape(features['direction'], [-1, 78, 78], name='direction_input_tensor')
        contrast_input_layer = tf.reshape(features['contrast'], [-1, 78, 78], name='contrast_input_tensor')

    # bounding box of the previous frame
    prev_bbox_input_layer = tf.reshape(features['prev_bbox'], [-1, 4])

    # Omid's MT model
    speed_tun_layer = SpeedTuning(n_channel,
                                  params,
                                  100./50 * 25, name='speed_tunning')([speed_input_layer, contrast_input_layer])
    direction_tun_layer = DirectionTuning(n_channel,
                                          params,
                                          name='direction_tunning')(direction_input_layer)

    with tf.variable_scope("mt_activity"):
        mt_feat = tf.multiply(speed_tun_layer, direction_tun_layer, name='MT_act_tensor')
        mt_feat = tf.where(tf.is_nan(mt_feat), tf.zeros_like(mt_feat), mt_feat)

    # visual tracking network
    mt_feat_flat = tf.reshape(mt_feat, [-1, 78 * 78 * n_channel], name='flattened_MT_act_tensor')
    concat = tf.concat([mt_feat_flat, prev_bbox_input_layer], axis=1)

    dense = tf.layers.dense(inputs=concat, units=4)

    # reshape it two 2 coordinates
    predictions = tf.reshape(dense, [-1, 2, 2], name='bbox_tensor')

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_or_create_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)