from os import path

import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io as sio


class PairwiseVideoFrameInputFunc(object):

    def __init__(self, dataset_dir,
                 batch_size=64,
                 num_epochs=-1,
                 shuffle_buffer_size=1024):
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.num_epochs = num_epochs
        self.shuffle_bsize = shuffle_buffer_size

    def _read_features_pyfunc(self, feat_dir, prev_bbox, bbox):
        opt_flow_filepath = path.join(feat_dir.decode(), 'optic_flow.mat')
        contrast_filepath = path.join(feat_dir.decode(), 'contrast.npy')

        # read optic flow
        opt_flow = sio.loadmat(opt_flow_filepath)
        h_flow = opt_flow['h_flow']
        v_flow = opt_flow['v_flow']

        # read contrast
        contrast = np.load(contrast_filepath)

        # cast to float32
        h_flow = h_flow.astype(np.float32)
        v_flow = v_flow.astype(np.float32)
        contrast = contrast.astype(np.float32)
        prev_bbox = prev_bbox.astype(np.float32)
        bbox = bbox.astype(np.float32)

        return h_flow, v_flow, contrast, prev_bbox, bbox

    def _parse(self, h_flow, v_flow, contrast, prev_bbox, bbox):
        h_flow.set_shape([78, 78])
        v_flow.set_shape([78, 78])
        contrast.set_shape([78, 78])
        prev_bbox.set_shape([2, 2])
        bbox.set_shape([2, 2])

        # convert cart to polar coordinates
        speed = tf.sqrt(tf.square(h_flow) + tf.square(v_flow))
        direction = tf.atan(v_flow / h_flow)
        direction = tf.where(tf.is_nan(direction), tf.zeros_like(direction), direction)

        return {'speed': speed, 'direction': direction, 'contrast': contrast, 'prev_bbox': prev_bbox}, bbox

    def __call__(self):
        pairwise_video_frame_df = pd.read_json(path.join(self.dataset_dir, 'index.json'))

        feat_dirs = pairwise_video_frame_df['feature_dir']\
            .map(lambda d: path.join(self.dataset_dir, d)).tolist()
        f1_bboxes = pairwise_video_frame_df['box1'].tolist()
        f2_bboxes = pairwise_video_frame_df['box2'].tolist()

        # parse the input
        pairwise_frame_dataset = tf.data.Dataset.from_tensor_slices((feat_dirs, f1_bboxes, f2_bboxes))
        pairwise_frame_dataset = pairwise_frame_dataset.map(lambda feat_dir, prev_bbox, bbox: tuple(tf.py_func(
            self._read_features_pyfunc,
            [feat_dir, prev_bbox, bbox],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])))
        pairwise_frame_dataset = pairwise_frame_dataset.map(self._parse)

        # make iterator
        pairwise_frame_dataset = pairwise_frame_dataset.shuffle(buffer_size=self.shuffle_bsize)
        pairwise_frame_dataset = pairwise_frame_dataset.batch(self.batch_size)
        pairwise_frame_dataset = pairwise_frame_dataset.repeat(self.num_epochs)

        iterator = pairwise_frame_dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        print('intput pipline initilized, input_shpae=%s, input_dtype=%s' %
              (pairwise_frame_dataset.output_shapes, pairwise_frame_dataset.output_types))

        return features, labels