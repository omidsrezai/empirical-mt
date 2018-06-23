from os import path

from skimage import io, img_as_float32, draw, img_as_ubyte
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io as sio
import cv2


class PairwiseVideoFrameInputFuncBase(object):

    def __init__(self, dataset_dir,
                 batch_size=64,
                 fixed_frame_size=76,
                 num_epochs=-1,
                 shuffle_buffer_size=1024):
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.num_epochs = num_epochs
        self.shuffle_bsize = shuffle_buffer_size
        self.fixed_frame_size = fixed_frame_size

    def _read_features_pyfunc(self, feat_dir, prev_bbox, bbox):
        opt_flow_filepath = path.join(feat_dir.decode(), 'optic_flow.mat')
        contrast_filepath = path.join(feat_dir.decode(), 'contrast.npy')
        prev_frame_filepath = path.join(feat_dir.decode(), 'prev_frame.png')
        frame_filepath = path.join(feat_dir.decode(), 'frame.png')

        # read optic flow
        opt_flow = sio.loadmat(opt_flow_filepath)
        h_flow_lk = opt_flow['h_flow'].astype(np.float32)
        v_flow_lk = opt_flow['v_flow'].astype(np.float32)

        # read contrast
        contrast = np.load(contrast_filepath).astype(np.float32)

        # read frames
        prev_frame = img_as_float32(io.imread(prev_frame_filepath, as_gray=True))
        frame = img_as_float32(io.imread(frame_filepath, as_gray=True))

        # calculate optic flow using Farneback
        prev_frame_u8 = img_as_ubyte(prev_frame)
        frame_u8 = img_as_ubyte(frame)
        opt_flow_farneback = cv2.calcOpticalFlowFarneback(prev_frame_u8, frame_u8, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        h_flow_farneback = opt_flow_farneback[:, :, 1]
        v_flow_farneback = opt_flow_farneback[:, :, 0]

        # make a heat map of previous bounding box
        prev_bbox_heat_map = np.zeros((self.fixed_frame_size, self.fixed_frame_size), dtype=np.float32)
        rr, cc = draw.rectangle(prev_bbox[0], prev_bbox[1])
        prev_bbox_heat_map[rr, cc] = 1.


        # cast bounding box to float32
        prev_bbox = prev_bbox.astype(np.float32)
        bbox = bbox.astype(np.float32)


        return h_flow_farneback, v_flow_farneback, h_flow_lk, v_flow_lk, contrast, prev_bbox, bbox, prev_frame, frame, prev_bbox_heat_map

    # abstract method

    def _parse(self, h_flow_f, v_flow_f, h_flow_lk, v_flow_lk, contrast, prev_bbox, bbox, prev_frame, frame, prev_bbox_heat_map):
        raise NotImplementedError

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
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])))
        pairwise_frame_dataset = pairwise_frame_dataset.map(self._parse)

        # make iterator
        # pairwise_frame_dataset = pairwise_frame_dataset.shuffle(buffer_size=self.shuffle_bsize)
        pairwise_frame_dataset = pairwise_frame_dataset.batch(self.batch_size)
        pairwise_frame_dataset = pairwise_frame_dataset.repeat(self.num_epochs)

        iterator = pairwise_frame_dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        print('intput pipline initilized, input_shpae=%s, input_dtype=%s' %
              (pairwise_frame_dataset.output_shapes, pairwise_frame_dataset.output_types))

        return features, labels



class SpeedDirectionInputFunc(PairwiseVideoFrameInputFuncBase):

    def _parse(self, h_flow_farneback, v_flow_farneback, _1, _2, contrast, prev_bbox, bbox, prev_frame, frame, _):
        h_flow_farneback.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        v_flow_farneback.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        contrast.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        prev_bbox.set_shape([2, 2])
        bbox.set_shape([2, 2])

        prev_frame.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        prev_frame = tf.reshape(prev_frame, [self.fixed_frame_size, self.fixed_frame_size, 1])
        frame.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        frame = tf.reshape(frame, [self.fixed_frame_size, self.fixed_frame_size, 1])

        # convert cart to polar coordinates
        speed = tf.sqrt(tf.square(h_flow_farneback) + tf.square(v_flow_farneback))
        direction = tf.atan(v_flow_farneback / h_flow_farneback)
        direction = tf.where(tf.is_nan(direction), tf.zeros_like(direction), direction)

        return {'speed': speed,
                'direction': direction,
                'contrast': contrast,
                'prev_bbox': prev_bbox,
                'prev_frame': prev_frame,
                'frame': frame}, bbox


class OptFlowContrastInputFunc(PairwiseVideoFrameInputFuncBase):

    def _parse(self, h_flow_f, v_flow_f, h_flow_lk, v_flow_lk, contrast, prev_bbox, bbox, prev_frame, frame, prev_bbox_heat_map):
        h_flow_f.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        v_flow_f.set_shape([self.fixed_frame_size, self.fixed_frame_size])

        h_flow_lk.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        v_flow_lk.set_shape([self.fixed_frame_size, self.fixed_frame_size])

        contrast.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        prev_frame.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        frame.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        prev_bbox.set_shape([2, 2])
        bbox.set_shape([2, 2])

        prev_frame = tf.reshape(prev_frame, [self.fixed_frame_size, self.fixed_frame_size, 1])
        frame = tf.reshape(frame, [self.fixed_frame_size, self.fixed_frame_size, 1])

        return {'h_flow_f': h_flow_f,
                'v_flow_f': v_flow_f,
                'h_flow_lk': h_flow_lk,
                'v_flow_lk': v_flow_lk,
                'contrast': contrast,
                'prev_bbox': prev_bbox,
                'prev_frame': prev_frame,
                'frame': frame,
                'prev_bbox_heat_map': prev_bbox_heat_map}, bbox


class SearchPatchObjPatchInputFn(PairwiseVideoFrameInputFuncBase):

    def _parse(self, _1, _2, _3, prev_bbox, bbox, object_patch, search_patch, _4):
        object_patch.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        search_patch.set_shape([self.fixed_frame_size, self.fixed_frame_size])
        prev_bbox.set_shape([2, 2])
        bbox.set_shape([2, 2])

        object_patch = tf.reshape(object_patch, [self.fixed_frame_size, self.fixed_frame_size, 1])
        search_patch = tf.reshape(search_patch, [self.fixed_frame_size, self.fixed_frame_size, 1])

        return {'prev_bbox': prev_bbox,
                'object_patch': object_patch,
                'search_patch': search_patch}, bbox