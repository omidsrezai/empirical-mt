import tensorflow as tf
from keras import backend as K
from keras.constraints import NonNeg

from surround.smart_example import SmartInput as TentLinearComb, AddBiasNonlinear, SmartConv2D as SelConv2d
from tuning import SpeedTuning, DirectionTuning
from visual_tracking.utils.keras_utils import NonPos

FIXED_FRAME_SIZE = 76 # TODO make this a parameter

tf_sess = tf.Session()
K.set_session(tf_sess)

class AreaMT(object):

    def __init__(self, speed_scalar, n_chann, empirical_excitatory_params, max_im_outputs):
        self.speed_scalar = speed_scalar
        self.excitatory_params = empirical_excitatory_params
        self.n_chann = n_chann
        self.max_im_outputs = max_im_outputs

    def __call__(self, speed_input, speed_input_tents, direction_input, contrast_input=None):

        with tf.variable_scope("area_mt"):
            '''
            excitatory component
            '''

            with tf.name_scope('sp_tun_excit'):
                speed_tun_excit_layer = SpeedTuning(self.n_chann,
                                                    self.excitatory_params,
                                                    self.speed_scalar,
                                                    name='sp_tun_excit')
                speed_tun_excit = speed_tun_excit_layer(speed_input)
                self._3d_tensor_summary(speed_tun_excit, 'speed_tun_excitatory')

            with tf.name_scope('dir_tun_excit'):
                direction_tun_excit_layer = DirectionTuning(self.n_chann,
                                                            self.excitatory_params,
                                                            name='dir_tun_excit')
                direction_tun_excit = direction_tun_excit_layer(direction_input)

                self._3d_tensor_summary(direction_tun_excit, 'dir_tun_excitatory')

            with tf.variable_scope("excitatory"):
                excitatory = tf.multiply(speed_tun_excit, direction_tun_excit)
                excitatory = tf.layers.batch_normalization(excitatory)
                excitatory = self._15x15_chann_sel_conv2d(excitatory, k_constraint=NonNeg())

                for t, n in zip([speed_tun_excit, direction_tun_excit, excitatory],
                                ['speed_tun_excit', 'direction_tun_excit', 'excitatory']):
                    self._3d_tensor_summary(t, n)

            '''
            direction-selective suppression component
            '''

            with tf.variable_scope('sp_tun_dir_sel_sup'):
                speed_tun_dir_sel_sup = TentLinearComb(self.n_chann,
                                                       constraint=NonNeg(),
                                                       name='sp_tun_dir_sel_sup')(speed_input_tents)

                self._3d_tensor_summary(speed_tun_dir_sel_sup, 'speed_tun_dir_sel_sup')

            # shares direction tunning with excitatory component
            direction_tun_dir_sel_sup = direction_tun_excit

            with tf.variable_scope("dir_sel_sup"):
                dir_selective_sup = tf.multiply(speed_tun_dir_sel_sup, direction_tun_dir_sel_sup)
                dir_selective_sup = tf.layers.batch_normalization(dir_selective_sup)
                dir_selective_sup = self._15x15_chann_sel_conv2d(dir_selective_sup, k_constraint=NonPos())

                for t, n in zip([speed_tun_dir_sel_sup, direction_tun_dir_sel_sup, dir_selective_sup],
                                ['speed_tun_dir_sel_sup', 'direction_tun_dir_sel_sup', 'direction_selective_sup']):
                    self._3d_tensor_summary(t, n)

            '''
            non-direction-selective suppression component
            '''

            with tf.variable_scope("sp_tun_non_dir_sup"):
                speed_tun_non_dir_sel_sup = TentLinearComb(self.n_chann,
                                                           constraint=NonNeg(),
                                                           name='sp_tun_non_dir_sup')(speed_input_tents)

                self._3d_tensor_summary(speed_tun_non_dir_sel_sup, 'speed_tun_non_dir_sup')

                # a fix to 'outbound_nodes'
                speed_tun_non_dir_sel_sup = tf.identity(speed_tun_non_dir_sel_sup)

            with tf.variable_scope("non_dir_sel_sup"):
                non_dir_sel_sup = tf.layers.batch_normalization(speed_tun_non_dir_sel_sup)
                non_dir_sel_sup = self._15x15_chann_sel_conv2d(non_dir_sel_sup, k_constraint=NonPos())

                for t, n in zip([speed_tun_non_dir_sel_sup, non_dir_sel_sup],
                                ['speed_tun_non_dir_sel_sup', 'non_direction_selective_sup']):
                    self._3d_tensor_summary(t, n)

            '''
            combine three components
            '''
            with tf.variable_scope('integrate'):
                mt_activity = excitatory + dir_selective_sup + non_dir_sel_sup
                mt_activity = AddBiasNonlinear(self.n_chann,
                                               activation='relu',
                                               use_bias=True,
                                               name='center')(mt_activity)
                mt_activity = tf.identity(mt_activity) # a fix to AddBiadNonLinear missing outbound_nodes
                mt_activity = tf.layers.conv2d(mt_activity,
                                               kernel_size=(15, 15),
                                               filters=self.n_chann,
                                               strides=(1, 1),
                                               padding='same',
                                               activation=tf.nn.relu)
                mt_activity = tf.layers.batch_normalization(mt_activity)
                mt_activity = tf.layers.max_pooling2d(mt_activity,
                                                      pool_size=(6, 6),
                                                      strides=(6, 6),
                                                      name='6x6_max_pool')

                self._3d_tensor_summary(mt_activity, 'area_mt_activity')

        return mt_activity

    def _3d_tensor_summary(self, tensor, name):
        tf.summary.histogram(name, tensor)
        tf.summary.image('%s_l2_norm' % name,
                         tf.norm(tensor, ord=2, axis=3, keep_dims=True),
                         max_outputs=self.max_im_outputs)

    def _15x15_chann_sel_conv2d(self, x, k_constraint):
        conv2d = SelConv2d(self.n_chann,
                           (15, 15),
                           activation=None,
                           use_bias=False,
                           padding="SAME",
                           kernel_constraint=k_constraint,
                           name='channel_sel_conv2d')

        y = conv2d(x)
        return y