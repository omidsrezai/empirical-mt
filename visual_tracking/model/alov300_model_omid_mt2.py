import pickle

import tensorflow as tf
from keras import  backend as K
import numpy as np

# link keras session and tensorflow session
import MT
from tuning import SpeedTuning, DirectionTuning
from visual_tracking.model.alov300_model_base import ALOV300ModelBase

LEARNING_RATE = 0.0001 # was 0.0001

MAX_IMG_OUTPUTS = 64

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)


class MTTracker(ALOV300ModelBase):

    def __init__(self, mt_params_path, speed_scaler, **kwargs):
        self.mt_params = pickle.load(open(mt_params_path, "rb"))
        self.n_chann = 64
        self.speed_scaler = speed_scaler
        super(MTTracker, self).__init__(**kwargs)

    def __call__(self, features, labels, mode):

        with tf.name_scope('frame_pairs'):
            with tf.name_scope('previous_frame'):
                self._summary_images_with_bbox(features['frame1'], features['box'], name='frame1_box')

            with tf.name_scope('frame'):
                self._summary_images_with_bbox(features['frame2'], labels, name='frame2_box')

        with tf.name_scope('speed_input'):
            speed_input = tf.identity(features['speed'])

            tf.summary.histogram('speed_input', speed_input)
            tf.summary.image('speed', tf.expand_dims(speed_input, axis=3), max_outputs=MAX_IMG_OUTPUTS)

        with tf.name_scope('direction_input'):
            direction_input = tf.identity(features['direction'])

            tf.summary.histogram('direction_input', direction_input)
            tf.summary.image('direction', tf.expand_dims(direction_input, axis=3), max_outputs=MAX_IMG_OUTPUTS)

        with tf.name_scope('mt_tunning_layers'):
            speed_tun_layer = SpeedTuning(64, self.mt_params, self.speed_scaler, name='speed_tunning')
            direction_tun_layer = DirectionTuning(64, self.mt_params, name='direction_tunning')

            speed_tun_layer = speed_tun_layer([speed_input, tf.zeros_like(speed_input)])
            direction_tun_layer = direction_tun_layer(direction_input)

            mt_activity = tf.multiply(speed_tun_layer, direction_tun_layer, name='mt_act_tensor')

            # visualize speeding tuning and direction tuning
            tf.summary.histogram("speed_tuning", speed_tun_layer)
            tf.summary.histogram("direction_tuning", direction_tun_layer)
            tf.summary.histogram('activations', mt_activity)

            tf.summary.image('speed_tun_l2_norm',
                             tf.norm(speed_tun_layer, ord=2, axis=3, keep_dims=True),
                             max_outputs=MAX_IMG_OUTPUTS)
            tf.summary.image('direction_tun_l2_norm',
                             tf.norm(direction_tun_layer, ord=2, axis=3, keep_dims=True),
                             max_outputs=MAX_IMG_OUTPUTS)
            tf.summary.image('mt_act_2_norm',
                             tf.norm(mt_activity, ord=2, axis=3, keep_dims=True),
                             max_outputs=MAX_IMG_OUTPUTS)

        mt_activity = tf.layers.batch_normalization(mt_activity, name='mt_act_batch_norm')
        tf.summary.image('batch_normed_mt_act', tf.norm(mt_activity, ord=2, axis=3, keep_dims=True))

        conv = self._conv2d(mt_activity,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=32,
                            padding='same',
                            name='conv1',
                            batch_norm=True)

        with tf.name_scope('attention_mask'):
            masks = tf.expand_dims(features['mask'], 3)
            tf.summary.image('attention_mask', masks, max_outputs=MAX_IMG_OUTPUTS)

            masked = tf.concat([conv, masks], axis=3)

        conv = self._conv2d(masked,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=32,
                            padding='valid',
                            name='conv2',
                            batch_norm=True)

        conv = tf.layers.average_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))

        conv = self._conv2d(conv,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            filters=32,
                            padding='valid',
                            name='conv3',
                            batch_norm=True)

        pool = tf.layers.average_pooling2d(conv, pool_size=(2, 2), strides=(2, 2))

        tf.summary.image('pool_l2_norm', tf.norm(pool, ord=2, axis=3, keep_dims=True), max_outputs=MAX_IMG_OUTPUTS)
        tf.summary.histogram('pool', pool)

        pool_flatten = tf.layers.flatten(pool)
        # pool_flatten = tf.layers.dropout(pool_flatten, 0.1)

        dense = self._dense(pool_flatten, units=1024, name='dense1', act=tf.nn.elu)
        dense = self._dense(dense, units=256, name='dense2')
        pbbox = self._dense(dense, units=4, name='dense3', act=tf.nn.sigmoid)

        self._summary_images_with_bbox(features['frame2'], pbbox, 'frame2_predicted_box')
        self._predicted_delta_summary(prev_bbox=features['box'], p_bbox=pbbox, t_bbox=labels)

        bbox_loss = tf.losses.mean_squared_error(labels=pbbox, predictions=labels)

        # TRAIN mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.minimize(
                loss=bbox_loss,
                global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, train_op=train_op)

        # EVAL mode
        eval_metrics_ops = self._get_eval_metrics_ops(predictions=pbbox - features['box'],
                                                      labels=labels - features['box'])

        return tf.estimator.EstimatorSpec(mode=mode, loss=bbox_loss, eval_metric_ops=eval_metrics_ops)

    def _flow_decoding(self, conv, direction_input, speed_input):

        with tf.name_scope('conv_supervise'):
            flow_input_x = tf.multiply(speed_input, tf.cos(direction_input))
            flow_input_y = tf.multiply(speed_input, tf.sin(direction_input))

            tf.summary.image('flow_x_input_l1', tf.expand_dims(tf.abs(flow_input_x), axis=3),
                             max_outputs=MAX_IMG_OUTPUTS)
            tf.summary.image('flow_y_input_l1', tf.expand_dims(tf.abs(flow_input_y), axis=3),
                             max_outputs=MAX_IMG_OUTPUTS)
            tf.summary.image('conv3_batch_normed_l2_norm', tf.norm(conv, ord=2, axis=3, keep_dims=True),
                             max_outputs=MAX_IMG_OUTPUTS)

            tf.summary.histogram('flow_input_x', flow_input_x)
            tf.summary.histogram('flow_input_y', flow_input_y)
            tf.summary.histogram('conv3_batch_norm', conv)

            conv_loss = tf.losses.mean_squared_error(labels=tf.stack([flow_input_y, flow_input_x], axis=3),
                                                     predictions=conv)
            tf.summary.scalar('conv_loss', conv_loss)
        return conv_loss, flow_input_x, flow_input_y

    def _get_eval_metrics_ops(self, labels, predictions):

        _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']
        pred_mean = tf.reduce_mean(predictions, axis=0)

        metrics_ops = {
            'l1_loss': tf.metrics.mean_absolute_error(labels=labels, predictions=predictions),
            'l2_loss': tf.metrics.mean_squared_error(labels=labels, predictions=predictions),
        }

        for i, v in enumerate(_dim_ids):
            metrics_ops['%s_mean' % v] = tf.metrics.mean(predictions[:, i])
            metrics_ops['%s_var' % v] = tf.metrics.mean(tf.square(predictions[:, i] - pred_mean[i]))
            metrics_ops['%s_covar' % v] = tf.contrib.metrics.streaming_covariance(predictions=predictions[:, i],
                                                                                  labels=labels[:, i])
            metrics_ops['%s_corr' % v] = tf.contrib.metrics.streaming_pearson_correlation(predictions=predictions[:, i],
                                                                                          labels=labels[:, i])

        return metrics_ops

    def _summary_images_with_bbox(self, images, boxes, name):

        boxes_3d = tf.stack([boxes, boxes, boxes], axis=1)
        annotated_im = tf.image.draw_bounding_boxes(images, boxes_3d)
        tf.summary.image(name, annotated_im, max_outputs=MAX_IMG_OUTPUTS)

    def _predicted_delta_summary(self, prev_bbox, p_bbox, t_bbox):

        _dim_ids = ['y_min', 'x_min', 'y_max', 'x_max']

        with tf.name_scope('prediction_metrics/delta'):
            p_delta = p_bbox - prev_bbox
            t_delta = t_bbox - prev_bbox

            t_mean, t_var = tf.nn.moments(t_delta, axes=0)
            p_mean, p_var = tf.nn.moments(p_delta, axes=0)

            covar = tf.reduce_mean(tf.multiply(p_delta, t_delta), axis=0) - tf.multiply(t_mean, p_mean)
            corr = tf.div(covar, tf.sqrt(tf.multiply(t_var, p_var)))

            for i, v in enumerate(_dim_ids):
                tf.summary.histogram('t_%s_delta' % v, t_delta[:, i])
                tf.summary.histogram('p_%s_delta' % v, p_delta[:, i])

                tf.summary.scalar('t_%s_delta_mean' % v, t_mean[i])
                tf.summary.scalar('p_%s_delta_mean' % v, p_mean[i])

                tf.summary.scalar('t_%s_delta_var' % v, t_var[i])
                tf.summary.scalar('p_%s_delta_var' % v, p_var[i])

                tf.summary.scalar('corvar_delta_%s' % v, covar[i])
                tf.summary.scalar('corr_delta_%s' % v, corr[i])