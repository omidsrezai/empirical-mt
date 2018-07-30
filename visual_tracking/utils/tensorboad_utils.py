from os import path

import tensorflow as tf


def feature_maps_summary(name, tensor, max_im_outputs):
    tf.summary.histogram(name, tensor)
    tf.summary.image('%s_l2_norm' % name,
                     tf.norm(tensor, ord=2, axis=3, keep_dims=True),
                     max_outputs=max_im_outputs)


def mt_conv_kernels_summary():
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