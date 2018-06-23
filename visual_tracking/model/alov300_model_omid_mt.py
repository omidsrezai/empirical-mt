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


class ALOV300OmideMTModelFn(object):

    def __init__(self, mt_params_path):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = 64

        self._tanh_dense_layer_count = 0

    def __call__(self, features, labels, mode):
        self._tanh_dense_layer_count = 0

        if mode == tf.estimator.ModeKeys.TRAIN:
            self._draw_input_frames(features, labels)

        contrast_input_layer, direction_input_layer, speed_input_layer = self._mt_inputs(features)

        # bounding box of the previous frame
        with tf.variable_scope('prev_bounding_box'):
            prev_bbox_input_layer = tf.reshape(features['prev_bbox'], [-1, 4])
            prev_bbox_input_layer = prev_bbox_input_layer / (FIXED_FRAME_SIZE - 1)

        # Omid's MT model
        # mt_act = self._mt_model(contrast_input_layer, direction_input_layer, speed_input_layer)

        # visual tracking network
        with tf.variable_scope('tracker_nn') as scope:

            '''
            conv1 = tf.layers.conv2d(mt_act, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)
            conv1 = tf.layers.conv2d(conv1, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(3, 3))
            pool1 = tf.layers.dropout(pool1, rate=0.2)

            pool1_image = tf.reshape(tf.norm(pool1, ord=64, axis=3), [-1, 25, 25, 1])
            tf.summary.image('pool1_images', pool1_image, max_outputs=10)


            conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                     activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv2, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=(3, 3), strides=(3, 3))
            pool2 = tf.layers.dropout(pool2, rate=0.2)


            pool2_image = tf.reshape(tf.norm(pool2, ord=64, axis=3), [-1, 8, 8, 1])
            tf.summary.image('pool2_images', pool2_image, max_outputs=10)

            # mt_act_max = tf.reduce_max(mt_act, axis=3)

            mt_act_flat = tf.layers.flatten(pool2, name='flattened_mt_act_tensor')
            tf.summary.histogram('mt_pool2_features', pool2)
            # mt_act_flat = tf.layers.dropout(mt_act_flat, rate=0.3)
            '''

            speed_input_layer = tf.reshape(speed_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            direction_input_layer = tf.reshape(direction_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            conv_speed = tf.layers.conv2d(speed_input_layer, filters=16, kernel_size=(5, 5), strides=(5, 5), activation=tf.nn.relu)
            # conv_speed = tf.layers.conv2d(conv_speed, filters=32, kernel_size=(3, 3), strides=(3, 3), activation=tf.nn.relu)
            conv_direction = tf.layers.conv2d(direction_input_layer, filters=16, kernel_size=(5, 5), strides=(5, 5), activation=tf.nn.relu)
            # conv_direction = tf.layers.conv2d(conv_direction, filters=32, kernel_size=(3, 3), strides=(3, 3), activation=tf.nn.relu)

            conv_speed = tf.layers.dropout(conv_speed)
            conv_direction = tf.layers.dropout(conv_direction)

            mt_act_flat = tf.layers.flatten(tf.concat([conv_speed, conv_direction], axis=3))

            dense = self._tanh_dense_layer(mt_act_flat, 1024)
            dense = self._tanh_dense_layer(dense, 256)
            dense = self._tanh_dense_layer(dense, 128)
            dense = self._tanh_dense_layer(dense, 8)

            concat = tf.concat([dense, prev_bbox_input_layer], axis=1)

            dense = self._tanh_dense_layer(concat, 8)

            dense = tf.layers.dense(dense, 4)
            tf.summary.histogram('before_sigmoid', dense)
            dense = tf.nn.sigmoid(dense)
            tf.summary.histogram('after_sigmoid', dense)

            self._draw_output_bbox(features['frame'], dense)

            # reshape bounding box to 2x2
            output = tf.reshape(dense, [-1, 2, 2], name='bbox_scaled')

            # plot weight matrix
            #scope.reuse_variables()
            #dense_kernel = tf.get_variable('dense_layer/kernel')
            #dense_kernel_mt_w = tf.reduce_mean(tf.abs(tf.slice(dense_kernel, [0, 0], [5776, 4])))
            #dense_kernel_prev_bbox_w = tf.reduce_mean(tf.abs(tf.slice(dense_kernel, [5776, 0], [4, 4])))
            #tf.summary.scalar('prev_bbox_over_mt_w', dense_kernel_prev_bbox_w / dense_kernel_mt_w)

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            # rescale it to coordinates
            predictions = {
                'bbox': output * (FIXED_FRAME_SIZE - 1)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=output * (FIXED_FRAME_SIZE - 1))

        with tf.variable_scope('ground_truth_bbox'):
            labels_rescaled = labels / (FIXED_FRAME_SIZE - 1)

        with tf.variable_scope('metrics'):
            p_offsets_in_pixels = features['prev_bbox'] - (output * (FIXED_FRAME_SIZE - 1))
            p_offsets_mean = tf.reduce_mean(p_offsets_in_pixels)
            p_offsets_var = tf.reduce_mean(tf.square(p_offsets_in_pixels - p_offsets_mean))
            tf.summary.scalar('predicted_offset_mean', p_offsets_mean)
            tf.summary.scalar('predicted_offset_var', p_offsets_var)

            t_offsets_in_pixels = features['prev_bbox'] - labels
            t_offsets_mean = tf.reduce_mean(t_offsets_in_pixels)
            t_offset_var = tf.reduce_mean(tf.square(t_offsets_in_pixels - t_offsets_mean))

            offsets_covar = tf.reduce_mean((t_offsets_in_pixels - t_offsets_mean) *
                                           (p_offsets_in_pixels - p_offsets_mean))
            tf.summary.scalar('true_offset_mean', t_offsets_mean)
            tf.summary.scalar('true_offset_var', t_offset_var)
            tf.summary.scalar('offsets_covariance', offsets_covar)

        loss = tf.losses.absolute_difference(labels=labels_rescaled, predictions=output)

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


    def _mt_model(self, contrast_input_layer, direction_input_layer, speed_input_layer):
        with tf.variable_scope('mt_model'):

            speed_tun_layer = SpeedTuning(64, self.mt_params, 2, name='speed_tunning')
            direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')
            speed_tun_layer = speed_tun_layer([speed_input_layer, contrast_input_layer])
            direction_tun_layer = direction_tun_layer(direction_input_layer)

            # visualize speeding tuning and direction tuning
            tf.summary.histogram("speed_tuning", speed_tun_layer)
            speed_tun_layer_images = tf.reshape(tf.norm(speed_tun_layer, ord=64, axis=3),
                                                [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            tf.summary.image('speed_tun', speed_tun_layer_images, max_outputs=10)

            tf.summary.histogram("direction_tuning", direction_tun_layer)
            # direction_tun_images = tf.reshape(tf.reduce_mean(direction_tun_layer, axis=3),
                                              # [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            # tf.summary.image('direction_tun', direction_tun_images, max_outputs=10)


            with tf.variable_scope("mt_activity"):
                mt_act = tf.multiply(speed_tun_layer, direction_tun_layer, name='mt_act_tensor')
                tf.summary.histogram('activations', mt_act)
                mt_act_image = tf.reshape(tf.norm(mt_act, ord=64, axis=3), [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
                tf.summary.image('mt_activities_mean', mt_act_image, max_outputs=10)

        return mt_act


    def _tanh_dense_layer(self, input, units):

        self._tanh_dense_layer_count += 1

        with tf.variable_scope('dense_layer_%s' % self._tanh_dense_layer_count):
            dense = tf.layers.dense(inputs=input, units=units)
            tf.summary.histogram('before_activation', dense)
            dense = tf.nn.tanh(dense)
            tf.summary.histogram('after_activation', dense)

        return dense


    def _mt_inputs(self, features):
        # input layers
        with tf.variable_scope("mt_inputs"):
            # speed input
            speed_input_layer = tf.identity(features['speed'], name='speed_input_tensor')
            speed_input_images = tf.reshape(speed_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            tf.summary.image('speed_input', speed_input_images, max_outputs=10)
            tf.summary.histogram('speed_input_original', speed_input_layer)
            tf.summary.histogram('speed_input_after', speed_input_layer * (100. / 50 * 5))

            tf.summary.scalar('speed_min', tf.reduce_min(speed_input_layer))
            tf.summary.scalar('speed_max', tf.reduce_max(speed_input_layer))
            tf.summary.scalar('speed_median', tf.contrib.distributions.percentile(speed_input_layer, 50))

            # direction input
            direction_input_layer = tf.identity(features['direction'], name='direction_input_tensor')

            # contrast input
            contrast_input_layer = tf.identity(features['contrast'], name='contrast_input_tensor')
            contrast_input_images = tf.reshape(contrast_input_layer, [-1, FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, 1])
            tf.summary.image('contrast_input', contrast_input_images, max_outputs=10)
            tf.summary.histogram('contrast_input', contrast_input_layer)

        return contrast_input_layer, direction_input_layer, speed_input_layer


    def _draw_input_frames(self, inputs, labels):
        with tf.variable_scope('input_frames'):
            prev_frame = inputs['prev_frame']
            prev_bbox = tf.reshape(inputs['prev_bbox'], [-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)
            frame = inputs['frame']
            ground_truth_bbox = tf.reshape(labels, shape=[-1, 1, 4]) / (FIXED_FRAME_SIZE - 1)

            prev_frame_bbox = tf.image.draw_bounding_boxes(prev_frame, prev_bbox)
            frame_bbox = tf.image.draw_bounding_boxes(frame, ground_truth_bbox)

            tf.summary.image("frame1", prev_frame_bbox, max_outputs=10)
            tf.summary.image("frame2", frame_bbox, max_outputs=10)


    def _draw_output_bbox(self, frame, bbox):
        with tf.variable_scope('predicted_bbox'):
            bbox = tf.reshape(bbox, [-1, 1, 4])
            frame_bbox = tf.image.draw_bounding_boxes(frame, bbox)
            tf.summary.image("predicted_bbox", frame_bbox, max_outputs=10)
