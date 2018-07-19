import sys

import cv2
import numpy as np
import tensorflow as tf
from skimage import transform, img_as_ubyte, color, util

from visual_tracking.data_pipline.sequence_input_pip import SequenceInputFuncBase
from visual_tracking.utils.optic_flow import compute_optic_flow
from visual_tracking.utils.saliency import compute_saliency


class SpeedDirectionSeqInputFunc(SequenceInputFuncBase):

    def __init__(self, mode=tf.estimator.ModeKeys.TRAIN,
                 flow_method='fb',
                 k=2,
                 fixed_input_dim=200,
                 speed_scalar=4,
                 **kwargs):
        self.mode = mode
        self.k = k
        self.fixed_input_dim = fixed_input_dim
        self.flow_method = flow_method
        self.speed_scalar = speed_scalar
        super(SpeedDirectionSeqInputFunc, self).__init__(**kwargs)

    def parse(self, dataset):
        dataset = dataset.map(lambda frames, bboxes:
                                tuple(tf.py_func(
                                    self._rescale_on_box1,
                                    [frames, bboxes],
                                    [tf.float32, tf.float32])),
                                num_parallel_calls=self.n_workers)\
            .map(lambda frames, bboxes:
                              tuple(tf.py_func(
                                  self._compute_optic_flow_and_saliency_map,
                                  [frames, bboxes],
                                  [tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers) \
            .map(lambda frames, flows, saliency, bboxes:
                              tuple(tf.py_func(
                                  self._pad_and_crop_to_box1,
                                  [frames, flows, saliency, bboxes],
                                  [tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=self.n_workers) \
            .map(lambda frames, flows, sailency, bboxes:
                 tuple(tf.py_func(
                     self._draw_attention_mask,
                     [frames, flows, sailency, bboxes],
                     [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])),
                 num_parallel_calls=self.n_workers)\
            .map(self._fmt_input, num_parallel_calls=self.n_workers)

        return dataset

    def _rescale_on_box1(self, frames, bboxes):
        # scale is computed so that k * max(obj_h, obj_w) = fix_dim
        _, frame_h, frame_w, _ = frames.shape

        y_min, x_min, y_max, x_max = np.multiply(bboxes[0, :],
                                                 np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))
        object_height = y_max - y_min
        object_width = x_max - x_min

        scale = self.fixed_input_dim / (self.k * max(object_height, object_width))

        frames_rescaled = []

        for i, f in enumerate(frames):
            frames_rescaled.append(transform.rescale(f, scale=scale, order=1, anti_aliasing=True).astype(np.float32))

        frames_rescaled = np.stack(frames_rescaled, axis=0)

        return frames_rescaled, bboxes

    def _compute_optic_flow_and_saliency_map(self, frames, bboxes):
        flows = []
        saliency = []

        for i in range(0, frames.shape[0] - 1):
            flows.append(compute_optic_flow(frames[i], frames[i+1], method=self.flow_method))
            saliency.append(compute_saliency(frames[i]))

        flows = np.stack(flows, axis=0)
        saliency = np.stack(saliency, axis=0)

        return frames, flows, saliency, bboxes

    def _pad_and_crop_to_box1(self, frames, flows, saliency, bboxes):
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
        saliency = np.expand_dims(saliency, axis=3)
        saliency_padded = _pad(saliency)

        bboxes_in_pixels = bboxes_in_pixels + paddings

        # crop to box1
        y_min, x_min, y_max, x_max = bboxes_in_pixels[0, :]
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
        saliency_cropped = _crop(saliency_padded)

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

        return frames_cropped, flows_cropped, np.squeeze(saliency_cropped), bbox_cropped

    def _draw_attention_mask(self, frames, flows, saliency, bboxes):
        y_min, x_min, y_max, x_max = np.round(bboxes[0, :] * (self.fixed_input_dim - 1)).astype(np.int32)

        mask = np.zeros_like(frames[0, :, :, 0], dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = 1

        return frames, flows, saliency, mask, bboxes

    def _fmt_input(self, frames, flows, saliency, mask, bboxes):
        frames.set_shape([self.max_seq_len, self.fixed_input_dim, self.fixed_input_dim, 3])
        flows.set_shape([self.max_seq_len - 1, self.fixed_input_dim, self.fixed_input_dim, 2])
        saliency.set_shape([self.max_seq_len - 1, self.fixed_input_dim, self.fixed_input_dim])
        mask.set_shape([self.fixed_input_dim, self.fixed_input_dim])
        bboxes.set_shape([self.max_seq_len, 4])

        # computes speed and direction from opticflow
        flow_x = flows[:, :, :, 0]
        flow_y = flows[:, :, :, 1]

        speed = tf.norm(flows, ord=2, axis=3)
        direction = tf.atan2(flow_y, flow_x)

        # project speed with tent basiss
        speed_tents_ts = []
        tent_centers = np.exp(np.arange(0, 5, .45))
        for i in range(0, speed.shape[0]):
            tent_basis = []
            for j in range(0, len(tent_centers) - 2):
                _left = tent_centers[j]
                _center = tent_centers[j + 1]
                _right = tent_centers[j + 2]

                x = speed[i] * self.speed_scalar
                y_left = (x - _left) / (_center - _left)
                y_right = (_right - x) / (_right - _center)

                y = tf.where((x >= _left) & (x <= _center), y_left, tf.zeros_like(x)) \
                    + tf.where((x >= _center) & (x <= _right), y_right, tf.zeros_like(x))

                tent_basis.append(y)

            speed_tents = tf.stack(tent_basis, axis=2)
            speed_tents_ts.append(speed_tents)

        speed_tents_ts = tf.stack(speed_tents_ts, axis=0)

        return {
            'frames': frames,
            'speed': speed,
            'speed_tents': speed_tents_ts,
            'saliency': saliency,
            'direction': direction,
            'mask': mask,
            'bbox': bboxes[0, :]
        }, bboxes[-1, :]