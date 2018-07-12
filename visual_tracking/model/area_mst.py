import tensorflow as tf
from keras import backend as K

from visual_tracking.model.neural_net_blocks import conv2d
from visual_tracking.utils.tensorboad_utils import feature_maps_summary

tf_sess = tf.Session()
K.set_session(tf_sess)

class AreaMST(object):

    def __init__(self, n_chann, max_im_outputs=64):
        self.max_im_outputs = max_im_outputs
        self._allocated = False
        self.n_chann = n_chann

    def __call__(self, mt_activity):
        with tf.variable_scope("area_mst", reuse=self._allocated):
            conv = mt_activity
            for id, n_filters in zip([1, 2], [64, 64]):
                conv = conv2d(conv,
                              kernel_size=(9, 9),
                              strides=(1, 1),
                              filters=self.n_chann,
                              padding='same',
                              name='mst_layer%s' % id,
                              batch_norm=True,
                              act=tf.nn.relu,
                              dropout=0.1,
                              kernel_l2_reg_scale=0.,
                              max_pool=None,
                              k_init_uniform=False)

            mst_activity = tf.layers.max_pooling2d(conv,
                                                   pool_size=(3, 3),
                                                   strides=(3, 3))

            feature_maps_summary('mst_activity',
                                 mst_activity,
                                 max_im_outputs=self.max_im_outputs)

        self._allocated = True

        return mst_activity