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
                 fixed_frame_size=76,
                 num_epochs=-1,
                 shuffle_buffer_size=8196,
                 n_workers=10):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_bsize = shuffle_buffer_size
        self.fixed_frame_size = fixed_frame_size
        self.dataset_index_filepath = dataset_index_filepath
        self.input_path = input_path
        self.n_workers = n_workers

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
            .shuffle(buffer_size=self.shuffle_bsize)\
            .map(lambda f1_filepath, f2_filepath, box1, box2:
                                            tuple(tf.py_func(
                                                self._read_frames_pyfunc,
                                                [f1_filepath, f2_filepath, box1, box2],
                                                [tf.float32, tf.float32, tf.float32, tf.float32])),
                 num_parallel_calls=self.n_workers)\

        pairwise_dataset = self.parse(pairwise_dataset)

        print(pairwise_dataset)

        # make iterator
        # pairwise_frame_dataset = pairwise_frame_dataset.shuffle(buffer_size=self.shuffle_bsize)
        pairwise_dataset = pairwise_dataset.batch(self.batch_size).repeat(self.num_epochs)

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


class EstimatorOpticFlowInputFunc(PairwiseInputFuncBase):

    def __init__(self, mode=tf.estimator.ModeKeys.TRAIN, k=2, fixed_input_dim=200, **kwargs):
        self.mode = mode
        self.k = k
        self.fixed_input_dim = fixed_input_dim
        super(EstimatorOpticFlowInputFunc, self).__init__(**kwargs)

    def parse(self, dataset):
        dataset = dataset.map(lambda frame1, frame2, box1, box2:
                              tuple(tf.py_func(
                                  self._rescale_on_box1,
                                  [frame1, frame2, box1, box2],
                                  [tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers)\
            .map(lambda frame1, frame2, box1, box2:
                              tuple(tf.py_func(
                                  self._compute_optic_flow,
                                  [frame1, frame2, box1, box2],
                                  [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers)\
            .map(lambda frame1, frame2, flow, box1, box2:
                              tuple(tf.py_func(
                                  self._pad_and_crop_to_box1,
                                  [frame1, frame2, flow, box1, box2],
                                  [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers)

        return dataset


    def _rescale_on_box1(self, frame1, frame2, box1, box2):
        frame_h, frame_w, _ = frame1.shape

        y_min, x_min, y_max, x_max = np.multiply(box1, np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))
        object_height = y_max - y_min
        object_width = x_max - x_min

        scale = self.fixed_input_dim / (self.k * max(object_height, object_width))

        frame1_rescaled = transform.rescale(frame1, scale=scale, order=2, anti_aliasing=True).astype(np.float32)
        frame2_rescaled = transform.rescale(frame2, scale=scale, order=2, anti_aliasing=True).astype(np.float32)

        return frame1_rescaled, frame2_rescaled, box1, box2

    def _compute_optic_flow(self, frame1, frame2, box1, box2):
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

        return frame1, frame2, flow_farneback, box1, box2

    def _pad_and_crop_to_box1(self, frame1, frame2, flow, box1, box2):
        frame_h, frame_w, _ = flow.shape

        box1_pixel = np.multiply(box1, np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))
        box2_pixel = np.multiply(box2, np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))

        # pad the boundary of frames
        paddings = self.fixed_input_dim

        def _pad(im):
            padded = util.pad(im, ((paddings, paddings), (paddings, paddings), (0 ,0)), mode='constant')
            return padded

        frame1_padded = _pad(frame1)
        frame2_padded = _pad(frame2)
        flow_padded = _pad(flow)
        box1_pixel = box1_pixel + paddings
        box2_pixel = box2_pixel + paddings

        # crop to box1
        y_min, x_min, y_max, x_max = box1_pixel
        object_h = y_max - y_min
        object_w = x_max - x_min
        y_center = (y_min + y_max) / 2.
        x_center = (x_min + x_max) / 2.

        y_start = np.round(y_center - (self.fixed_input_dim / 2.)).astype(np.int32)
        x_start = np.round(x_center - (self.fixed_input_dim / 2.)).astype(np.int32)
        y_end = y_start + self.fixed_input_dim
        x_end = x_start + self.fixed_input_dim

        def _crop(im):
            cropped = im[y_start:y_end, x_start:x_end, :]
            return cropped

        frame1_cropped = _crop(frame1_padded)
        frame2_cropped = _crop(frame2_padded)
        flow_cropped = _crop(flow_padded)

        # re-caculate box1, box2
        y_center_cropped = self.fixed_input_dim / 2.
        x_center_cropped = self.fixed_input_dim / 2.
        y_start_cropped = y_center_cropped - (object_h / 2.)
        x_start_cropped = x_center_cropped - (object_w / 2.)

        box1_cropped = np.array([y_start_cropped, x_start_cropped, y_start_cropped + object_h, x_start_cropped + object_w])
        box2_cropped = box1_cropped + (box2_pixel - box1_pixel)

        box1_cropped = box1_cropped / (self.fixed_input_dim - 1)
        box2_cropped = box2_cropped / (self.fixed_input_dim - 1)

        box1_cropped = box1_cropped.astype(np.float32)
        box2_cropped = box2_cropped.astype(np.float32)

        return frame1_cropped, frame2_cropped, flow_cropped, box1_cropped, box2_cropped



if __name__ == '__main__':
    ne = EstimatorOpticFlowInputFunc(dataset_index_filepath='../../data/alov300++/alov300_train.csv', input_path='../../', batch_size=10)()

    with tf.Session() as sess:
        r = sess.run(ne)
        print(r)