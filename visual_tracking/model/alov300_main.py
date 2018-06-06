import tensorflow as tf


def nn_model_fn(features, labels, mode):
    mt_feat_input_layer = tf.reshape(features['frame'], [-1, 78, 78, 1])
    mt_feat_input_layer = tf.cast(mt_feat_input_layer, tf.float32)
    prev_bbox_input_layer = tf.reshape(features['prev_bbox'], [-1, 4])
    prev_bbox_input_layer = tf.cast(prev_bbox_input_layer, dtype=tf.float32)

    mt_feat_flat = tf.reshape(mt_feat_input_layer, [-1, 78 * 78])
    concat = tf.concat([mt_feat_flat, prev_bbox_input_layer], axis=1)

    dense = tf.layers.dense(inputs=concat, units=4)

    # reshape it two 2 coordinates
    predictions = tf.reshape(dense, [-1, 2, 2])

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