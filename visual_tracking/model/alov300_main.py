import tensorflow as tf
from keras import  backend as K
import numpy as np

# link keras session and tensorflow session
import MT
from tuning import SpeedTuning, DirectionTuning

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

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
    _draw_input_frames(features, labels)

    contrast_input_layer, direction_input_layer, speed_input_layer = _mt_inputs(features)

    # bounding box of the previous frame
    with tf.variable_scope('prev_bounding_box'):
        prev_bbox_input_layer = tf.reshape(features['prev_bbox'], [-1, 4])
        prev_bbox_input_layer = prev_bbox_input_layer / (FIXED_FRAME_SIZE - 1)

    # Omid's MT model
    mt_act = _mt_model(contrast_input_layer, direction_input_layer, speed_input_layer)

    # visual tracking network
    with tf.variable_scope('tracker_nn'):
        mt_act_flat = tf.reshape(mt_act, [-1, FIXED_FRAME_SIZE * FIXED_FRAME_SIZE * n_channel], name='flattened_mt_act_tensor')
        concat = tf.concat([mt_act_flat, prev_bbox_input_layer], axis=1)

        dense = tf.layers.dense(inputs=concat, units=4, name='dense_layer')
        tf.summary.histogram('dense_layer_before_nonlinear', dense)
        dense = tf.nn.sigmoid(dense)
        tf.summary.histogram('dense_layer_nonlinear', dense)

        _draw_output_bbox(features['frame'], dense)

        # reshape bounding box to 2x2
        output = tf.reshape(dense, [-1, 2, 2], name='bbox_scaled')

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        # rescale it to coordinates
        predictions = output * (FIXED_FRAME_SIZE - 1)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.variable_scope('ground_truth_bbox'):
        labels_rescaled = labels / (FIXED_FRAME_SIZE - 1)

    loss = tf.losses.absolute_difference(labels=labels_rescaled, predictions=output)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_or_create_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def _mt_model(contrast_input_layer, direction_input_layer, speed_input_layer):
    with tf.variable_scope('mt_model'):

        speed_tun_layer = SpeedTuning(n_channel,
                                      params,
                                      100. / 50 * 25 * 5, name='speed_tunning')([speed_input_layer, contrast_input_layer])
        direction_tun_layer = DirectionTuning(n_channel,
                                              params,
                                              name='direction_tunning')(direction_input_layer)

        # visualize speeding tuning and direction tuning
        speed_tun_layer_images = tf.reshape(tf.reduce_mean(speed_tun_layer, axis=3),
                                            [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
        tf.summary.image('speed_tun', speed_tun_layer_images)

        direction_tun_images = tf.reshape(tf.reduce_mean(direction_tun_layer, axis=3),
                                          [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
        tf.summary.image('direction_tun', direction_tun_images)


        with tf.variable_scope("mt_activity"):
            mt_act = tf.multiply(speed_tun_layer, direction_tun_layer, name='mt_act_tensor')
            mt_act = tf.where(tf.is_nan(mt_act), tf.zeros_like(mt_act), mt_act)

            tf.summary.histogram('activations', mt_act)

            mt_act_mean = tf.reduce_mean(mt_act, axis=3)
            mt_act_mean = tf.reshape(mt_act_mean, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            tf.summary.image('mt_activities_mean', mt_act_mean)

    return mt_act


def _mt_inputs(features):
    # input layers
    with tf.variable_scope("mt_inputs"):
        # speed input
        speed_input_layer = tf.identity(features['speed'], name='speed_input_tensor')
        speed_input_images = tf.reshape(speed_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
        tf.summary.image('speed_input', speed_input_images)
        tf.summary.histogram('speed_input', speed_input_layer)

        # direction input
        direction_input_layer = tf.identity(features['direction'], name='direction_input_tensor')

        # contrast input
        contrast_input_layer = tf.identity(features['contrast'], name='contrast_input_tensor')
        contrast_input_images = tf.reshape(contrast_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
        tf.summary.image('contrast_input', contrast_input_images)
        tf.summary.histogram('contrast_input', contrast_input_layer)

    return contrast_input_layer, direction_input_layer, speed_input_layer


def _draw_input_frames(inputs, labels):
    with tf.variable_scope('input_frames'):
        prev_frame = inputs['prev_frame']
        prev_bbox = tf.reshape(inputs['prev_bbox'], [-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)
        frame = inputs['frame']
        ground_truth_bbox = tf.reshape(labels, shape=[-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)

        prev_frame_bbox = tf.image.draw_bounding_boxes(prev_frame, prev_bbox)
        frame_bbox = tf.image.draw_bounding_boxes(frame, ground_truth_bbox)

        tf.summary.image("frame1", prev_frame_bbox)
        tf.summary.image("frame2", frame_bbox)


def _draw_output_bbox(frame, bbox):
    with tf.variable_scope('predicted_bbox'):
        bbox = tf.reshape(bbox, [-1, 1, 4])
        frame_bbox = tf.image.draw_bounding_boxes(frame, bbox)
        tf.summary.image("predicted_bbox", frame_bbox)
