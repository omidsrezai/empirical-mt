import csv
from os import path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, img_as_float32, draw, img_as_ubyte, color, transform, util


class PairwiseInputFuncBase(object):

    def __init__(self,
                 dataset_index_filepath,
                 input_path='../',
                 batch_size=64,
                 num_epochs=-1,
                 shuffle_buffer_size=500,
                 n_workers=10,
                 prefetch_buffer_size=300):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_bsize = shuffle_buffer_size,
        self.dataset_index_filepath = dataset_index_filepath
        self.input_path = input_path
        self.n_workers = n_workers
        self.prefetch_buffer_size=prefetch_buffer_size

    def _pairwise_group_frames_pyfunc(self, video_folderpath, annotation_filepath, height, width):

        # change to relative path
        video_folderpath_rel = path.join(self.input_path, video_folderpath)
        annotation_filepath_rel = path.join(self.input_path, annotation_filepath)

        frame1_ids = []
        frame2_ids = []
        bounding_boxes1 = np.array([], dtype=np.float32).reshape((0, 4))
        bounding_boxes2 = np.array([], dtype=np.float32).reshape((0, 4))

        def _unpack_annotation_row(r):
            id = int(r[0])

            y_min = float(r[4])
            x_min = float(r[3])
            y_max = float(r[8])
            x_max = float(r[7])

            x_min, x_max = sorted((x_min, x_max))
            y_min, y_max = sorted((y_min, y_max))

            return id, y_min, x_min, y_max, x_max

        with open(annotation_filepath_rel, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            prev_frame_id = None

            for row in reader:
                if prev_frame_id is None:
                    prev_frame_id, prev_y_min, prev_x_min, prev_y_max, prev_x_max = _unpack_annotation_row(row)
                else:
                    frame_id, y_min, x_min, y_max, x_max = _unpack_annotation_row(row)
                    num_frames = frame_id - prev_frame_id + 1

                    # interpolate bounding boxes for the frames between annotated frames
                    _frame_ids = range(prev_frame_id, frame_id + 1)
                    y_mins = np.linspace(prev_y_min, y_min, num=num_frames, dtype=np.float32) / (height - 1)
                    x_mins = np.linspace(prev_x_min, x_min, num=num_frames, dtype=np.float32) / (width - 1)
                    y_maxs = np.linspace(prev_y_max, y_max, num=num_frames, dtype=np.float32) / (height - 1)
                    x_maxs = np.linspace(prev_x_max, x_max, num=num_frames, dtype=np.float32) / (width - 1)

                    _bounding_boxes = np.transpose(np.stack([y_mins, x_mins, y_maxs, x_maxs], axis=0))
                    _bounding_boxes1 = _bounding_boxes[0:(num_frames - 1), :]
                    _bounding_boxes2 = _bounding_boxes[1:num_frames, :]

                    # append results
                    frame1_ids.extend(_frame_ids[0:(num_frames - 1)])
                    frame2_ids.extend(_frame_ids[1:num_frames])
                    bounding_boxes1 = np.concatenate([bounding_boxes1, _bounding_boxes1], axis=0)
                    bounding_boxes2 = np.concatenate([bounding_boxes2, _bounding_boxes2], axis=0)

                    # update previous frame
                    prev_frame_id, prev_y_min, prev_x_min, prev_y_max, prev_x_max = (frame_id, y_min, x_min, y_max, x_max)

        # map frame id to frame file path
        frame1_filepaths = np.array([path.join(video_folderpath_rel, '%08d.jpg' % id) for id in frame1_ids])
        frame2_filepaths = np.array([path.join(video_folderpath_rel, '%08d.jpg' % id) for id in frame2_ids])

        return frame1_filepaths, frame2_filepaths, bounding_boxes1, bounding_boxes2

    def _read_frames_pyfunc(self, f1_filepath, f2_filepath, bounding_box1, bounding_box2):
        frame1 = img_as_float32(io.imread(f1_filepath))
        frame2 = img_as_float32(io.imread(f2_filepath))

        # convert gray scale frame to rgb
        if len(frame1.shape) == 2:
            frame1 = color.gray2rgb(frame1)
            frame2 = color.gray2rgb(frame2)

        return frame1, frame2, bounding_box1, bounding_box2

    def _compute_optic_flow_contrast_pyfunc(self, frame1, frame2):
        pass

    def parse(self, dataset):
        return dataset.map(self._resize_frames_720p)

    def _resize_frames_720p(self, frame1, frame2, box1, box2):
        frame1.set_shape([None, None, None])
        frame2.set_shape([None, None, None])

        frame1_resized = tf.image.resize_images(frame1, [720, 1280])
        frame2_resized = tf.image.resize_images(frame2, [720, 1280])

        return frame1_resized, frame2_resized, box1, box2

    def __call__(self):
        dataset_df = pd.read_csv(self.dataset_index_filepath)

        video_paths = dataset_df['video_path'].tolist()
        annotation_paths = dataset_df['ground_truth_path'].tolist()
        frame_heights = dataset_df['size_x'].tolist()
        frame_widths = dataset_df['size_y'].tolist()

        # parse the input
        pairwise_dataset = tf.data.Dataset.from_tensor_slices((video_paths, annotation_paths, frame_heights, frame_widths))
        print(pairwise_dataset)

        pairwise_dataset = pairwise_dataset\
            .flat_map(lambda video_path, annotation_path, h, w:
                                                 tf.data.Dataset.from_tensor_slices(tuple(tf.py_func(
                                                     self._pairwise_group_frames_pyfunc,
                                                     [video_path, annotation_path, h, w],
                                                     [tf.string, tf.string, tf.float32, tf.float32]))))\
            .map(lambda f1_filepath, f2_filepath, box1, box2:
                                            tuple(tf.py_func(
                                                self._read_frames_pyfunc,
                                                [f1_filepath, f2_filepath, box1, box2],
                                                [tf.float32, tf.float32, tf.float32, tf.float32])),
                 num_parallel_calls=self.n_workers) \

        # apply addtional pre-processing
        pairwise_dataset = self.parse(pairwise_dataset)

        pairwise_dataset = pairwise_dataset.cache(filename='../../tmp')\
            .repeat(self.num_epochs)\
            .batch(self.batch_size)\
            .prefetch(2)

        iterator = pairwise_dataset.make_one_shot_iterator()
        features = iterator.get_next()

        print('intput pipline initilized, input_shpae=%s, input_dtype=%s' %
              (pairwise_dataset.output_shapes, pairwise_dataset.output_types))

        return features


class AvgOpticFlowInputFunc(PairwiseInputFuncBase):

    def parse(self, dataset):
        dataset = dataset.map(lambda frame1, frame2, box1, box2:
                                            tuple(tf.py_func(
                                                self._compute_avg_optic_flow_pyfunc,
                                                [frame1, frame2, box1, box2],
                                                [tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers)
        return dataset

    def _compute_avg_optic_flow_pyfunc(self, frame1, frame2, box1, box2):
        # scale bounding boxes
        h, w, _ = frame1.shape
        box1 = np.multiply(box1, np.array([h - 1, w - 1, h - 1, w - 1])).astype(np.float32)
        box2 = np.multiply(box2, np.array([h - 1, w - 1, h - 1, w - 1])).astype(np.float32)

        # convert frames to uint8 gray scale
        f1_u8_gray = img_as_ubyte(color.rgb2gray(frame1))
        f2_u8_gray = img_as_ubyte(color.rgb2gray(frame2))

        flow_farneback = cv2.calcOpticalFlowFarneback(f1_u8_gray,
                                                      f2_u8_gray,
                                                      flow=None,
                                                      pyr_scale=0.5,
                                                      levels=4,
                                                      winsize=15,
                                                      iterations=3,
                                                      poly_n=5,
                                                      poly_sigma=1.1,
                                                      flags=0)

        flow_x = flow_farneback[:, :, 0].astype(np.float32)
        flow_y = flow_farneback[:, :, 1].astype(np.float32)

        # compute average of optic flow inside bounding box
        box_mask = np.zeros_like(flow_x, dtype=np.float32)
        y_min, x_min, y_max, x_max = np.round(box1).astype(np.int32)
        rr, cc = draw.rectangle((y_min, x_min), (y_max, x_max), shape=box_mask.shape)
        box_mask[rr, cc] = 1.

        avg_flow_in_x = np.sum(np.multiply(flow_x, box_mask)) / np.sum(box_mask)
        avg_flow_in_y = np.sum(np.multiply(flow_y, box_mask)) / np.sum(box_mask)

        center1 = (box1[0:2] + box1[2:4]) / 2
        center2 = (box2[0:2] + box2[2:4]) / 2

        return avg_flow_in_x, avg_flow_in_y, center2 - center1