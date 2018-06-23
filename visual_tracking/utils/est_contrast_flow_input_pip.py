import cv2
import numpy as np
import tensorflow as tf
from skimage import transform, img_as_ubyte, color, util

from visual_tracking.utils.pairwise_input_pip import PairwiseInputFuncBase


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
                              num_parallel_calls=self.n_workers) \
            .map(lambda frame1, frame2, flow, box1, box2:
                              tuple(tf.py_func(
                                  self._pad_and_crop_to_box1,
                                  [frame1, frame2, flow, box1, box2],
                                  [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers)\
            .map(self._fmt_input, num_parallel_calls=self.n_workers)

        return dataset


    def _rescale_on_box1(self, frame1, frame2, box1, box2):
        frame_h, frame_w, _ = frame1.shape

        y_min, x_min, y_max, x_max = np.multiply(box1, np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))
        object_height = y_max - y_min
        object_width = x_max - x_min

        scale = self.fixed_input_dim / (self.k * max(object_height, object_width))

        frame1_rescaled = transform.rescale(frame1, scale=scale, order=1, anti_aliasing=True).astype(np.float32)
        frame2_rescaled = transform.rescale(frame2, scale=scale, order=1, anti_aliasing=True).astype(np.float32)

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


    def _fmt_input(self, frame1, frame2, flow, box1, box2):
        frame1.set_shape([self.fixed_input_dim, self.fixed_input_dim, 3])
        frame2.set_shape([self.fixed_input_dim, self.fixed_input_dim, 3])
        flow.set_shape([self.fixed_input_dim, self.fixed_input_dim, 2])
        box1.set_shape([4])
        box2.set_shape([4])

        flow_x = tf.squeeze(tf.slice(flow, [0, 0, 0], [self.fixed_input_dim, self.fixed_input_dim, 1]))
        flow_y = tf.squeeze(tf.slice(flow, [0, 0, 1], [self.fixed_input_dim, self.fixed_input_dim, 1]))

        speed = tf.norm(flow, ord=2, axis=2)
        direction = tf.atan(flow_y / flow_x)
        direction = tf.where(tf.is_nan(direction), tf.zeros_like(direction), direction)

        return {
            'frame1': frame1,
            'frame2': frame2,
            'speed': speed,
            'direction': direction,
            'box': box1
        }, box2
