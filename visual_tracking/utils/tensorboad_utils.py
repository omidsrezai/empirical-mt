import tensorflow as tf


def feature_maps_summary(name, tensor, max_im_outputs):
    tf.summary.histogram(name, tensor)
    tf.summary.image('%s_l2_norm' % name,
                     tf.norm(tensor, ord=2, axis=3, keep_dims=True),
                     max_outputs=max_im_outputs)