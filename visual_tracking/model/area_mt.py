from os import path

import tensorflow as tf
from keras import backend as K
from keras.constraints import NonNeg

from surround.smart_example import SmartInput as TentLinearComb, AddBiasNonlinear, SmartConv2D as SelConv2d
from tuning import SpeedTuning, DirectionTuning
from visual_tracking.model.layer_tools import conv2d
from visual_tracking.utils.keras_utils import NonPos
from visual_tracking.utils.tensorboad_utils import feature_maps_summary

tf_sess = tf.Session()
K.set_session(tf_sess)


class AreaMT(object):

    def __init__(self, speed_scalar,
                 n_chann,
                 conv_chann,
                 empirical_excitatory_params,
                 attention_gains,
                 chann_sel_dp,
                 max_im_outputs,
                 activity_dp):
        self.speed_scalar = speed_scalar
        self.excitatory_params = empirical_excitatory_params
        self.n_chann = n_chann
        self.max_im_outputs = max_im_outputs
        self._allocated = False
        self.chann_sel_dp = chann_sel_dp
        self.activity_dp = activity_dp
        self.attention_gains = attention_gains[0:self.n_chann]
        self.conv_chann = conv_chann

    def __call__(self, speed_input, speed_input_tents, direction_input, saliency_input=None, contrast_input=None):
        with tf.variable_scope("area_mt", reuse=self._allocated):
            direction_tun = self._direction_tuning(direction_input)
            attention_tun = self._attention_tuning(saliency_input) if saliency_input is not None else None

            excitatory = self._excitatory(direction_tun, attention_tun, speed_input)
            dir_selective_sup = self._direction_selective_suppressive(direction_tun, attention_tun, speed_input_tents)
            non_dir_sel_sup = self._non_direction_selective_suppressive(speed_input_tents, attention_tun)

            mt_activity = self._integrate_components(dir_selective_sup, excitatory, non_dir_sel_sup)

            if (not self._allocated):
                self._log_conv_kernels()

        self._allocated = True # flags the variables have been allocated

        return mt_activity

    def _attention_tuning(self, saliencymap):
        with tf.name_scope('attention_tun'):
            saliencymap_repeats = tf.tile(tf.expand_dims(saliencymap, axis=3), [1, 1, 1, self.n_chann])

            attention_gain_shape = saliencymap.get_shape().as_list()[1:3] + [self.n_chann]
            attention_gains = tf.reshape(tf.constant(self.attention_gains), [1, 1, self.n_chann])
            attention_gains = tf.tile(attention_gains, attention_gain_shape)

            attention_tun = (saliencymap_repeats * attention_gains) + (1 - saliencymap_repeats)
            attention_tun = tf.layers.batch_normalization(attention_tun)

            feature_maps_summary('attentin_tunning',
                                 attention_tun,
                                 max_im_outputs=self.max_im_outputs)

        return attention_tun

    def _direction_tuning(self, direction_input):
        with tf.name_scope('direction_tun'):
            direction_tun_excit_layer = DirectionTuning(self.n_chann,
                                                        self.excitatory_params,
                                                        name='direction_tun')
            direction_tun = direction_tun_excit_layer(direction_input)

            feature_maps_summary('direction_tun',
                                 direction_tun,
                                 max_im_outputs=self.max_im_outputs)

        return direction_tun

    def _excitatory(self, direction_tun, attention_tun, speed_input):
        with tf.name_scope('sp_tun_excit'):
            speed_tun_excit_layer = SpeedTuning(self.n_chann,
                                                self.excitatory_params,
                                                self.speed_scalar,
                                                name='sp_tun_excit')
            speed_tun_excit = speed_tun_excit_layer(speed_input)

            feature_maps_summary('speed_tun_excitatory',
                                 speed_tun_excit,
                                 max_im_outputs=self.max_im_outputs)

        with tf.variable_scope("excitatory"):
            excitatory = tf.multiply(speed_tun_excit, direction_tun)

            if attention_tun is not None:
                excitatory = tf.multiply(excitatory, attention_tun)

            excitatory = tf.layers.batch_normalization(excitatory)
            excitatory = self._15x15_chann_sel_conv2d(excitatory,
                                                      k_constraint=NonNeg(),
                                                      dp=self.chann_sel_dp)

            feature_maps_summary('excitatory',
                                 excitatory,
                                 max_im_outputs=self.max_im_outputs)

        return excitatory

    def _direction_selective_suppressive(self, direction_tun, attention_tun, speed_input_tents):
        with tf.variable_scope('sp_tun_dir_sel_sup'):
            speed_tun_dir_sel_sup = TentLinearComb(self.n_chann,
                                                   constraint=NonNeg(),
                                                   name='sp_tun_dir_sel_sup')(speed_input_tents)

            feature_maps_summary('speed_tun_dir_sel_sup',
                                 speed_tun_dir_sel_sup,
                                 max_im_outputs=self.max_im_outputs)

        with tf.variable_scope("dir_sel_sup"):
            dir_selective_sup = tf.multiply(speed_tun_dir_sel_sup, direction_tun)

            if attention_tun is not None:
                dir_selective_sup = tf.multiply(dir_selective_sup, attention_tun)

            dir_selective_sup = tf.layers.batch_normalization(dir_selective_sup)
            dir_selective_sup = self._15x15_chann_sel_conv2d(dir_selective_sup,
                                                             k_constraint=NonPos(),
                                                             dp=self.chann_sel_dp)

            feature_maps_summary('direction_selective_sup',
                                 dir_selective_sup,
                                 max_im_outputs=self.max_im_outputs)

        return dir_selective_sup

    def _non_direction_selective_suppressive(self, attention_tun, speed_input_tents):
        with tf.variable_scope("sp_tun_non_dir_sup"):
            speed_tun_non_dir_sel_sup = TentLinearComb(self.n_chann,
                                                       constraint=NonNeg(),
                                                       name='sp_tun_non_dir_sup')(speed_input_tents)
            speed_tun_non_dir_sel_sup = tf.identity(speed_tun_non_dir_sel_sup) # a fix to outbound_nodes

            feature_maps_summary('speed_tun_non_dir_sup',
                                 speed_tun_non_dir_sel_sup,
                                 max_im_outputs=self.max_im_outputs)

        with tf.variable_scope("non_dir_sel_sup"):
            if attention_tun is not None:
                speed_tun_non_dir_sel_sup = tf.multiply(speed_tun_non_dir_sel_sup, attention_tun)

            non_dir_sel_sup = tf.layers.batch_normalization(speed_tun_non_dir_sel_sup)
            non_dir_sel_sup = self._15x15_chann_sel_conv2d(non_dir_sel_sup,
                                                           k_constraint=NonPos(),
                                                           dp=self.chann_sel_dp)

            feature_maps_summary('non_direction_selective_sup',
                                 non_dir_sel_sup,
                                 max_im_outputs=self.max_im_outputs)

        return non_dir_sel_sup

    def _integrate_components(self, dir_selective_sup, excitatory, non_dir_sel_sup):
        with tf.variable_scope('integrate'):
            mt_activity = excitatory + dir_selective_sup + non_dir_sel_sup

            with tf.variable_scope('relu_act'):
                mt_activity = AddBiasNonlinear(self.n_chann,
                                               activation='relu',
                                               use_bias=True,
                                               name='add_bias_relu')(mt_activity)
                mt_activity = tf.identity(mt_activity)  # a fix to AddBiadNonLinear missing outbound_nodes

            mt_activity = tf.layers.batch_normalization(mt_activity)

            mt_activity = conv2d(mt_activity,
                                 kernel_size=(15, 15),
                                 filters=self.conv_chann,
                                 strides=(1, 1),
                                 padding='same',
                                 act=tf.nn.relu,
                                 dropout=self.activity_dp,
                                 batch_norm=True,
                                 max_pool=(6, 6),
                                 k_init_uniform=True,
                                 name='conv2d_6x6_pool',
                                 kernel_summary=not self._allocated,
                                 activity_summary=True)

        return mt_activity

    def _log_conv_kernels(self):
        tenors_to_log = ['excitatory', 'dir_sel_sup', 'non_dir_sel_sup']
        sig2 = .09

        with tf.name_scope('kernel_visualization'):
            for t_path in tenors_to_log:
                kernel_path = path.join('mt_over_time/time_map_area_mt/area_mt',
                                        t_path,
                                        'chann_sel_conv2d/conv2d/kernel_smart:0')
                selector_path = path.join('mt_over_time/time_map_area_mt/area_mt',
                                          t_path,
                                          'chann_sel_conv2d/conv2d/selector:0')

                kernel = tf.get_default_graph().get_tensor_by_name(kernel_path)
                selector = tf.get_default_graph().get_tensor_by_name(selector_path)

                tf.summary.histogram('%s_chann_selector' % t_path, selector)
                tf.summary.histogram('%s_chann_sel_conv2d_kernel' % t_path, kernel)

                base = tf.range(0, kernel.get_shape().as_list()[-1], dtype=tf.float32)
                base = tf.expand_dims(base, axis=1)
                base = tf.tile(base, [1, kernel.shape[-1]])
                kernels = kernel * tf.exp(-tf.square(base - selector) / (2 * sig2))

                for i in range(kernel.shape[-1]):
                    w = kernels[:, :, :, i]
                    confidence = tf.abs(tf.reduce_mean(w, axis=(0, 1)))
                    selected = w[:, :, tf.argmax(confidence)]
                    normed = tf.abs(selected)
                    normed = (normed - tf.reduce_min(normed)) / (tf.reduce_max(normed) - tf.reduce_min(normed))
                    tf.summary.image('%s_out_chann_%s' % (t_path, i),
                                     tf.expand_dims(tf.expand_dims(normed, axis=2), axis=0))

    def _15x15_chann_sel_conv2d(self, x, k_constraint, dp=0.):
        with tf.variable_scope('chann_sel_conv2d'):
            conv2d = SelConv2d(self.conv_chann,
                               (15, 15),
                               activation=None,
                               use_bias=False,
                               padding="SAME",
                               kernel_constraint=k_constraint,
                               name='conv2d',
                               kernel_initializer='glorot_uniform')

            y = tf.identity(conv2d(x)) # fixes no out_bound bug

        if dp > 0.:
            y = tf.layers.dropout(y)

        return y