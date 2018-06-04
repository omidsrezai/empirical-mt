import tensorflow as tf
import pandas as pd
from skimage import io, transform, color, draw, img_as_float32
from os import path
import numpy as np


def _read_obj_patch_search_patch_ann_function(f1_filepath, f2_filepath, f1_ann, f2_ann):
    f1 = io.imread(path.join('../', f1_filepath.decode()), as_gray=True)  # TODO parametrize prefix
    f2 = io.imread(path.join('../', f2_filepath.decode()), as_gray=True)

    #f1 = img_as_float32(f1)
    #f2 = img_as_float32(f2)

    object_patch = transform.resize(f1, (50, 50))
    search_patch = transform.resize(f2, (100, 100))

    heat_map = np.zeros((100, 100), dtype=np.float64)
    rr, cc = draw.rectangle((20, 20), (50, 50))
    heat_map[rr, cc] = 1.
    heat_map = heat_map.flatten()

    # cast to float
    object_patch = img_as_float32(object_patch)
    search_patch = img_as_float32(search_patch)
    heat_map = heat_map.astype(np.float32)

    return object_patch, search_patch, heat_map
    # return {'object_patch': object_patch, 'search_patch': search_path}, dummy_label


def _parse(object_patch, search_patch, heat_map):
    return {'object_patch': object_patch, 'search_patch': search_patch}, heat_map


def siamese_dataset_input_fn():
    siamese_df = pd.read_json('../data/alov300++/siamese.json')

    f1_filepaths = siamese_df['f1_filepath'].tolist()
    f2_filepaths = siamese_df['f2_filepath'].tolist()
    f1_anns = siamese_df['f1_ann'].tolist()
    f2_anns = siamese_df['f2_ann'].tolist()

    siamese_dataset = tf.data.Dataset.from_tensor_slices((f1_filepaths, f2_filepaths, f1_anns, f2_anns))
    siamese_dataset = siamese_dataset.map(lambda f1_filepath, f2_filepath, f1_ann, f2_ann: tuple(tf.py_func(
        _read_obj_patch_search_patch_ann_function, [f1_filepath, f2_filepath, f1_ann, f2_ann],
        [tf.float32, tf.float32, tf.float32])))
    siamese_dataset = siamese_dataset.map(_parse)

    siamese_dataset = siamese_dataset.shuffle(buffer_size=1024)  # TODO parametrize buffer_size
    siamese_dataset = siamese_dataset.batch(32)   # TODO parametrize batch_size
    siamese_dataset = siamese_dataset.repeat(10)  # TODO parametrize num_epochs

    iterator = siamese_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def siamese_cnn_fn(features, labels, mode):
    object_patch_input_layer = tf.reshape(features['object_patch'], [-1, 50, 50, 1])
    search_patch_input_layer = tf.reshape(features['search_patch'], [-1, 100, 100, 1])

    obj_patch_conv1 = tf.layers.conv2d(inputs=object_patch_input_layer,
                                       filters=4,
                                       kernel_size=[3, 3],
                                       padding='same',
                                       activation=tf.nn.relu)

    search_patch_input_layer = tf.layers.conv2d(inputs=search_patch_input_layer,
                                                filters=2,
                                                kernel_size=[3, 3],
                                                padding='same',
                                                activation=tf.nn.relu)

    obj_patch_flat = tf.reshape(obj_patch_conv1, [-1, 4 * 50 * 50])
    search_patch_flat = tf.reshape(search_patch_input_layer, [-1, 2 * 100 * 100])

    concat = tf.concat([obj_patch_flat, search_patch_flat], axis=1)

    dense = tf.layers.dense(inputs=concat, units=10000)

    predictions = tf.nn.sigmoid(dense)
    predictions = tf.reshape(predictions, [-1, 100, 100, 1])

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=dense)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_or_create_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



if __name__ == '__main__':
    siamese_visual_tracker = tf.estimator.Estimator(model_fn=siamese_cnn_fn, model_dir='../models')

    siamese_visual_tracker.train(input_fn=siamese_dataset_input_fn,
                                 steps=1)


