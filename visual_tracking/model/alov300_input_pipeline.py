from os import path

import pandas as pd
import tensorflow as tf


class PairwiseVideoFrameInputFunc(object):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def _parse(object_patch, frame_filepath, prev_bbox, bbox):
        frame = tf.image.decode_png(tf.read_file(frame_filepath), dtype=tf.uint8)

        return {'frame': frame, 'prev_bbox': prev_bbox}, bbox

    def __call__(self):
        pairwise_video_frame_df = pd.read_json(path.join(self.dataset_dir, 'index.json'))

        f1_filepaths = pairwise_video_frame_df['feature_dir']\
            .map(lambda d: path.join(self.dataset_dir, d, 'frame.png')).tolist()
        f1_anns = pairwise_video_frame_df['box1'].tolist()
        f2_anns = pairwise_video_frame_df['box2'].tolist()

        pairwise_frame_dataset = tf.data.Dataset.from_tensor_slices((f1_filepaths, f1_anns, f2_anns))
        pairwise_frame_dataset = pairwise_frame_dataset.map(self._parse)

        pairwise_frame_dataset = pairwise_frame_dataset.shuffle(buffer_size=64)  # TODO parametrize buffer_size
        pairwise_frame_dataset = pairwise_frame_dataset.batch(32)   # TODO parametrize batch_size
        pairwise_frame_dataset = pairwise_frame_dataset.repeat(10)  # TODO parametrize num_epochs

        iterator = pairwise_frame_dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels