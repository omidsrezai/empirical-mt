import cv2
import numpy as np
import tensorflow as tf
from skimage import transform, img_as_ubyte, color, util

import sys
sys.path.insert(0, '/home/ading/dev/pyflow/')
import pyflow

from visual_tracking.utils.pairwise_input_pip import PairwiseInputFuncBase
from visual_tracking.utils.sequence_input_pip import SixFrameInputFuncBase


class SeqEstimatorOpticFlowInputFunc(SixFrameInputFuncBase):

    def __init__(self, mode=tf.estimator.ModeKeys.TRAIN, flow_method='c2f', k=2, fixed_input_dim=200, **kwargs):
        self.mode = mode
        self.k = k
        self.fixed_input_dim = fixed_input_dim
        self.flow_method = flow_method
        super(SeqEstimatorOpticFlowInputFunc, self).__init__(**kwargs)

    def parse(self, dataset):
        dataset = dataset.map(lambda frames, bboxes:
                                tuple(tf.py_func(
                                    self._rescale_on_box1,
                                    [frames, bboxes],
                                    [tf.float32, tf.float32])),
                                num_parallel_calls=self.n_workers)\
            .map(lambda frames, bboxes:
                              tuple(tf.py_func(
                                  self._compute_optic_flow,
                                  [frames, bboxes],
                                  [tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers) \
            .map(lambda frames, flows, bboxes:
                              tuple(tf.py_func(
                                  self._pad_and_crop_to_box1,
                                  [frames, flows, bboxes],
                                  [tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers) \
            .map(lambda frames, flows, bboxes:
                 tuple(tf.py_func(
                     self._draw_attention_mask,
                     [frames, flows, bboxes],
                     [tf.float32, tf.float32, tf.float32, tf.float32])),
                 num_parallel_calls=self.n_workers)\
            .map(self._fmt_input, num_parallel_calls=self.n_workers)

        return dataset


    def _rescale_on_box1(self, frames, bboxes):
        # scale is computed so that k * max(obj_h, obj_w) = fix_dim
        _, frame_h, frame_w, _ = frames.shape

        y_min, x_min, y_max, x_max = np.multiply(bboxes[0, :], np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))
        object_height = y_max - y_min
        object_width = x_max - x_min

        scale = self.fixed_input_dim / (self.k * max(object_height, object_width))

        frames_rescaled = []

        for i, f in enumerate(frames):
            frames_rescaled.append(transform.rescale(f, scale=scale, order=1, anti_aliasing=True).astype(np.float32))

        frames_rescaled = np.stack(frames_rescaled, axis=0)

        return frames_rescaled, bboxes

    def _compute_optic_flow(self, frames, bboxes):
        # convert frames to uint8 gray scale
        if self.flow_method == 'fb':
            def _compute_optic_flow(frame1, frame2):
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
                return flow

        elif self.flow_method == 'c2f':
            # Flow Options:
            alpha = 0.012
            ratio = 0.75
            minWidth = 20
            nOuterFPIterations = 7
            nInnerFPIterations = 1
            nSORIterations = 30
            colType = 0 #RGB

            def _compute_optic_flow(frame1, frame2):
                frame1_float64 = frame1.astype(np.double)
                frame2_float64 = frame2.astype(np.double)

                u, v, _ = pyflow.coarse2fine_flow(frame1_float64, frame2_float64,
                                                  alpha, ratio, minWidth,
                                                  nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

                flow = np.stack([u, v], axis=2).astype(np.float32)

                return flow

        else:
            raise ValueError('Unkown flow method %s' % self.flow_method)

        flows = []
        for i in range(0, frames.shape[0] - 1):
            flows.append(_compute_optic_flow(frames[i], frames[i+1]))
        flows = np.stack(flows, axis=0)

        return frames, flows, bboxes

    def _pad_and_crop_to_box1(self, frames, flows, bboxes):
        _, frame_h, frame_w, _ = frames.shape

        # convert bounding boxes into pixel uints
        bboxes_in_pixels = np.zeros_like(bboxes)

        for i, bbox in enumerate(bboxes):
            bboxes_in_pixels[i, :] = np.multiply(bbox, np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))

        # pad the boundary of frames
        paddings = self.fixed_input_dim

        def _pad(im):
            padded = util.pad(im, ((0, 0), (paddings, paddings), (paddings, paddings), (0 ,0)), mode='constant')
            return padded

        frames_padded = _pad(frames)
        flows_padded = _pad(flows)

        bboxes_in_pixels = bboxes_in_pixels + paddings

        # crop to box1
        y_min, x_min, y_max, x_max = bboxes_in_pixels[0, :]
        object_h = y_max - y_min
        object_h = y_max - y_min
        object_w = x_max - x_min
        y_center = (y_min + y_max) / 2.
        x_center = (x_min + x_max) / 2.

        y_start = np.round(y_center - (self.fixed_input_dim / 2.)).astype(np.int32)
        x_start = np.round(x_center - (self.fixed_input_dim / 2.)).astype(np.int32)
        y_end = y_start + self.fixed_input_dim
        x_end = x_start + self.fixed_input_dim

        def _crop(im):
            cropped = im[:, y_start:y_end, x_start:x_end, :]
            return cropped

        frames_cropped = _crop(frames_padded)
        flows_cropped = _crop(flows_padded)

        # re-calculate bounding boxes
        bbox_cropped = np.zeros_like(bboxes_in_pixels, dtype=np.float32)

        y_center_cropped = self.fixed_input_dim / 2.
        x_center_cropped = self.fixed_input_dim / 2.
        y_start_cropped = y_center_cropped - (object_h / 2.)
        x_start_cropped = x_center_cropped - (object_w / 2.)

        bbox_cropped[0, :] = np.array([y_start_cropped,
                                       x_start_cropped,
                                       y_start_cropped + object_h,
                                       x_start_cropped + object_w])

        for i, box in enumerate(bboxes_in_pixels):
            if i > 0:
                # bbox_cropped[i] - bbox_cropped[0] = bboxes[i] - bbox[0]
                bbox_cropped[i, :] = bbox_cropped[0, :] + (box - bboxes_in_pixels[0, :])

        bbox_cropped = bbox_cropped / (self.fixed_input_dim - 1) # scale it to 0 to 1

        return frames_cropped, flows_cropped, bbox_cropped

    def _draw_attention_mask(self, frames, flows, bboxes):
        y_min, x_min, y_max, x_max = np.round(bboxes[0, :] * (self.fixed_input_dim - 1)).astype(np.int32)

        mask = np.zeros_like(frames[0, :, :, 0], dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = 1

        return frames, flows, mask, bboxes

    def _fmt_input(self, frames, flows, mask, bboxes):
        frames.set_shape([self.max_seq_len, self.fixed_input_dim, self.fixed_input_dim, 3])
        flows.set_shape([self.max_seq_len - 1, self.fixed_input_dim, self.fixed_input_dim, 2])
        mask.set_shape([self.fixed_input_dim, self.fixed_input_dim])
        bboxes.set_shape([self.max_seq_len, 4])

        flow_x = flows[:, :, :, 0]
        flow_y = flows[:, :, :, 1]

        speed = tf.norm(flows, ord=2, axis=3)
        direction = tf.atan2(flow_y, flow_x)

        return {
            'frames': frames,
            'speed': speed,
            'direction': direction,
            'mask': mask,
            'bbox': bboxes[0, :]
        }, bboxes[-1, :]