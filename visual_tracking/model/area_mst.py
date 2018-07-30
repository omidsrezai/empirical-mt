import tensorflow as tf
from keras import backend as K

from visual_tracking.model.layer_tools import conv2d

tf_sess = tf.Session()
K.set_session(tf_sess)

class AreaMST(object):

    def __init__(self, n_chann, max_im_outputs=64, dropout=0., l2_reg_scale=0.):
        self.max_im_outputs = max_im_outputs
        self._allocated = False
        self.n_chann = n_chann
        self.l2_reg_scale = l2_reg_scale
        self.dropout = dropout

    def __call__(self, mt_activity):
        with tf.variable_scope("area_mst", reuse=self._allocated):
            mst_layer1 = conv2d(mt_activity,
                                kernel_size=(9, 9),
                                strides=(1, 1),
                                filters=self.n_chann,
                                padding='same',
                                name='mst_layer1',
                                batch_norm=True,
                                act=tf.nn.relu,
                                dropout=self.dropout,
                                kernel_l2_reg_scale=self.l2_reg_scale,
                                max_pool=None,
                                k_init_uniform=True,
                                kernel_summary=not self._allocated)

            mst_layer2 = conv2d(mst_layer1,
                                kernel_size=(9, 9),
                                strides=(1, 1),
                                filters=self.n_chann,
                                padding='same',
                                name='mst_layer2',
                                batch_norm=True,
                                act=tf.nn.relu,
                                dropout=self.dropout,
                                kernel_l2_reg_scale=self.l2_reg_scale,
                                max_pool=(3, 3),
                                k_init_uniform=True,
                                kernel_summary=not self._allocated)

        self._allocated = True

        return mst_layer2