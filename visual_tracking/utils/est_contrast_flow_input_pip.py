import cv2
import numpy as np
import tensorflow as tf
from skimage import transform, img_as_ubyte, color, util

import sys
sys.path.insert(0, '/home/ading/dev/pyflow/')
import pyflow

from visual_tracking.utils.pairwise_input_pip import PairwiseInputFuncBase


class EstimatorOpticFlowInputFunc(PairwiseInputFuncBase):

    def __init__(self, mode=tf.estimator.ModeKeys.TRAIN,
                 flow_method='c2f',
                 k=2,
                 fixed_input_dim=200,
                 speed_scalar=4,
                 **kwargs):

        self.mode = mode
        self.k = k
        self.fixed_input_dim = fixed_input_dim
        self.flow_method = flow_method
        super(EstimatorOpticFlowInputFunc, self).__init__(**kwargs)
        self.speed_scalar = speed_scalar

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
                              num_parallel_calls=self.n_workers) \
            .map(lambda frame1, frame2, flow, box1, box2:
                 tuple(tf.py_func(
                     self._draw_attention_mask,
                     [frame1, frame2, flow, box1, box2],
                     [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])),
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
        if self.flow_method == 'fb':
            f1_u8_gray = img_as_ubyte(color.rgb2gray(frame1))
            f2_u8_gray = img_as_ubyte(color.rgb2gray(frame2))

            flow = cv2.calcOpticalFlowFarneback(f1_u8_gray,
                                                          f2_u8_gray,
                                                          flow=None,
                                                          pyr_scale=0.5,
                                                          levels=4,
                                                          winsize=15,
                                                          iterations=3,
                                                          poly_n=5,
                                                          poly_sigma=1.1,
                                                          flags=0)
        elif self.flow_method == 'c2f':
            # Flow Options:
            alpha = 0.012
            ratio = 0.75
            minWidth = 20
            nOuterFPIterations = 7
            nInnerFPIterations = 1
            nSORIterations = 30
            colType = 0 #RGB

            frame1_float64 = frame1.astype(np.double)
            frame2_float64 = frame2.astype(np.double)

            u, v, _ = pyflow.coarse2fine_flow(frame1_float64, frame2_float64,
                                              alpha, ratio, minWidth,
                                              nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

            flow = np.stack([u, v], axis=2).astype(np.float32)

        else:
            raise ValueError('Unkown flow method %s' % self.flow_method)

        return frame1, frame2, flow, box1, box2

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

    def _draw_attention_mask(self, frame1, frame2, flow, box1, box2):
        y_min, x_min, y_max, x_max = np.round(box1 * (self.fixed_input_dim - 1)).astype(np.int32)

        mask = np.zeros_like(frame1[:, :, 0], dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = 1

        return frame1, frame2, flow, box1, box2, mask

    def _fmt_input(self, frame1, frame2, flow, box1, box2, mask):
        frame1.set_shape([self.fixed_input_dim, self.fixed_input_dim, 3])
        frame2.set_shape([self.fixed_input_dim, self.fixed_input_dim, 3])
        flow.set_shape([self.fixed_input_dim, self.fixed_input_dim, 2])
        mask.set_shape([self.fixed_input_dim, self.fixed_input_dim])
        box1.set_shape([4])
        box2.set_shape([4])

        flow_x = tf.squeeze(tf.slice(flow, [0, 0, 0], [self.fixed_input_dim, self.fixed_input_dim, 1]))
        flow_y = tf.squeeze(tf.slice(flow, [0, 0, 1], [self.fixed_input_dim, self.fixed_input_dim, 1]))

        speed = tf.norm(flow, ord=2, axis=2)
        direction = tf.atan2(flow_y, flow_x)

        # project speed into the space of tent basis
        tent_basis = []
        tent_centers = np.exp(np.arange(0, 5, .45))

        for i in range(0, len(tent_centers) - 2):
            _left = tent_centers[i]
            _center = tent_centers[i + 1]
            _right = tent_centers[i + 2]

            x = speed * self.speed_scalar
            y_left = (x - _left) / (_center - _left)
            y_right = (_right - x) / (_right - _center)

            y = tf.where((x >= _left) & (x <= _center), y_left, tf.zeros_like(x)) \
                + tf.where((x >= _center) & (x <= _right), y_right, tf.zeros_like(x))

            tent_basis.append(y)

        speed_tents = tf.stack(tent_basis, axis=2)

        return {
            'frame1': frame1,
            'frame2': frame2,
            'speed': speed,
            'speed_tents': speed_tents,
            'direction': direction,
            'box': box1,
            'mask': mask
        }, box2
