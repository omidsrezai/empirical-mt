import numpy as np
import tensorflow as tf
from skimage import transform, util

from visual_tracking.data_pipline.alov300_sequence_input_base import SequenceInputFuncBase
from visual_tracking.utils.optic_flow import compute_optic_flow
from visual_tracking.utils.saliency import compute_saliency


class SpeedDirectionSaliencySeqInputFunc(SequenceInputFuncBase):

    def __init__(self, mode=tf.estimator.ModeKeys.TRAIN,
                 flow_method='fb',
                 saliency_method='mb',
                 k=2,
                 fixed_input_dim=200,
                 speed_scalar=4,
                 optic_flow_cache_id=None,
                 saliency_map_cache_id=None,
                 **kwargs):
        self.mode = mode
        self.k = k
        self.fixed_input_dim = fixed_input_dim
        self.flow_method = flow_method
        self.speed_scalar = speed_scalar
        self.saliency_method = saliency_method
        self.optic_flow_cache_id = optic_flow_cache_id
        self.saliency_map_cache_id = saliency_map_cache_id
        super(SpeedDirectionSaliencySeqInputFunc, self).__init__(**kwargs)

    def preprocess_cached(self, dataset):
        # since all preprocess steps are CPU-bound python functions, multi-threading is disabled,
        # num_parallel_calls are set to 1
        dataset = dataset.map(lambda frames, bboxes, num_seqs:
                                tuple(tf.py_func(
                                    self._rescale_on_box1,
                                    [frames, bboxes, num_seqs],
                                    [tf.float32, tf.float32, tf.int32],
                                    name='rescale')),
                                num_parallel_calls=1)\
            .map(lambda frames, bboxes, num_seqs:
                              tuple(tf.py_func(
                                  self._compute_saliency_map,
                                  [frames, bboxes, num_seqs],
                                  [tf.float32, tf.float32, tf.float32, tf.int32],
                                  name='compute_saliency')),
                              num_parallel_calls=1) \
            .map(lambda frames, saliencymaps, bboxes, num_seqs:
                 tuple(tf.py_func(
                     self._compute_optic_flow,
                     [frames, saliencymaps, bboxes, num_seqs],
                     [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32],
                     name='compute_optic_flow')),
                 num_parallel_calls=1) \
            .map(lambda frames, flows, saliency, bboxes, num_seqs:
                              tuple(tf.py_func(
                                  self._pad_and_crop_to_box1,
                                  [frames, flows, saliency, bboxes, num_seqs],
                                  [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32],
                                  name='pad_and_crop')),
                              num_parallel_calls=1)

        return dataset

    def preprocess_no_cache(self, dataset):
        dataset = dataset.map(self._set_shapes, num_parallel_calls=self.n_workers)\
            .map(self._random_90_rotation, num_parallel_calls=self.n_workers)\
            .map(self._random_hflip, num_parallel_calls=self.n_workers)\
            .map(self._fmt_input, num_parallel_calls=self.n_workers)
        return dataset

    # rescale the frames based on the size of object to search in the first frame
    def _rescale_on_box1(self, frames, bboxes, num_seqs):
        # scale is computed so that k * max(obj_h, obj_w) = fix_dim in the first frame
        _, frame_h, frame_w, _ = frames.shape

        y_min, x_min, y_max, x_max = np.multiply(bboxes[0, :],
                                                 np.array([frame_h - 1, frame_w - 1, frame_h - 1, frame_w - 1]))
        object_height = y_max - y_min
        object_width = x_max - x_min

        scale = self.fixed_input_dim / (self.k * max(object_height, object_width))

        frames_rescaled = [transform.rescale(f, scale=scale, order=1, anti_aliasing=True).astype(np.float32)
                           for f in frames]

        frames_rescaled = np.stack(frames_rescaled, axis=0)

        return frames_rescaled, bboxes, num_seqs

    # computes the salinecy detection for all frames
    def _compute_saliency_map(self, frames, bboxes, num_seqs):
        # if saliencymaps are not loaded from disk
        if 'saliencymaps_folderpath' in self.kv_pairs:
            saliency = frames[:, :, :, -1]  # saliency maps are loaded upstream and concated as 4-th channel
            frames = frames[:, :, :, 0:3]
        else:
            saliency = [compute_saliency(fi, method=self.saliency_method) for fi in frames]
            saliency = np.stack(saliency, axis=0)

        return frames, saliency, bboxes, num_seqs

    # computes optic flow for all frames
    def _compute_optic_flow(self, frames, saliency, bboxes, num_seqs):
        flows = []
        for i in range(0, frames.shape[0] - 1):
            flows.append(compute_optic_flow(frames[i], frames[i+1], method=self.flow_method))
        flows = np.stack(flows, axis=0)

        return frames, flows, saliency, bboxes, num_seqs

    # crop all frames to fix_input_dim x fixed_input_dim at the same coordinates
    # so that the object to track is centered in the first frame
    def _pad_and_crop_to_box1(self, frames, flows, saliency, bboxes, num_seqs):
        _, frame_h, frame_w, _ = frames.shape

        # convert bounding boxes (scaled to 0 to 1) into pixel uints
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

        # re-calculate bounding boxes after cropping
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

        return frames_cropped, flows_cropped, np.squeeze(saliency_cropped), bbox_cropped, num_seqs

    def _set_shapes(self, frames, flows, saliency, bboxes, num_seqs):
        frames.set_shape([self.sequence_length, self.fixed_input_dim, self.fixed_input_dim, 3])
        flows.set_shape([self.sequence_length - 1, self.fixed_input_dim, self.fixed_input_dim, 2])
        saliency.set_shape([self.sequence_length - 1, self.fixed_input_dim, self.fixed_input_dim])
        bboxes.set_shape([self.sequence_length, 4])
        num_seqs.set_shape([])

        return frames, flows, saliency, bboxes, num_seqs

    def _random_90_rotation(self, frames, flows, saliency, bboxes, num_seqs):
        # random rotation of k*90 degrees counter-clockwise
        def _rot90_k(k):
            # _rot_fn will work with arbitrary rotations apart from multiplies of 90 degrees,
            # but it is restricted to multiples of 90 degrees at the moment
            def _rot_fn():
                frames_rot = tf.image.rot90(frames, k=k)
                flows_rot = tf.image.rot90(flows, k=k)
                saliency_rot = tf.squeeze(tf.image.rot90(tf.expand_dims(saliency, axis=3), k=k))

                # rotation matrix transposed
                rot_mat_t = tf.stack([tf.cos(k * (-np.pi / 2.)),
                                    tf.sin(k * (-np.pi / 2.)),
                                    -tf.sin(k * (-np.pi / 2.)),
                                    tf.cos(k * (-np.pi / 2.))])
                rot_mat_t = tf.cast(tf.reshape(rot_mat_t, [2, 2]), dtype=np.float32)

                # reshape bboxes into ((y_min, y_max), (x_min, x_max))
                bboxes_transpose = tf.transpose(tf.reshape(bboxes, [-1, 2, 2]), perm=(0, 2, 1))

                # rotate bounding boxes for every frame
                #  step 1: move origin to center of image: (y_m, x_m) = (y, x) - (0.5, 0.5)
                #  step 2: rotate: (y_r, x_r) = rot_mat_t * (y_m, x_m)^T
                #  step 3: move origin back to lower right corner: (y_r, x_r) + 0.5
                bboxes_rot_transpose = tf.map_fn(lambda x: tf.matmul(rot_mat_t, x - 0.5),
                                                 elems=bboxes_transpose,
                                                 parallel_iterations=1) + 0.5

                # reshape bounding boxes back to y_min, x_min, y_max, x_max
                bboxes_rot = tf.reshape(tf.transpose(bboxes_rot_transpose, perm=(0, 2, 1)), [-1, 4])

                bboxes_rot = tf.map_fn(self._reorder_coords_single_bbox, elems=bboxes_rot, parallel_iterations=1)

                return frames_rot, flows_rot, saliency_rot, bboxes_rot

            return _rot_fn

        k = tf.random_uniform(shape=[], minval=0, maxval=4, dtype=np.int32)

        rotated = tf.case({tf.equal(k, 0): _rot90_k(0),
                          tf.equal(k, 1): _rot90_k(1),
                          tf.equal(k, 2): _rot90_k(2),
                          tf.equal(k, 3): _rot90_k(3)})

        # unpack
        frames_rot, flows_rot, saliency_rot, bboxes_rot = rotated

        return frames_rot, flows_rot, saliency_rot, bboxes_rot, num_seqs

    # random horizontal flip
    def _random_hflip(self, frames, flows, saliency, bboxes, num_seqs):
        # draw from a bernouli distribution
        do_hflip = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)

        def _no_flip():
            # identity
            return frames, flows, saliency, bboxes

        def _hflip():
            frames_flipped = tf.reverse(frames, axis=[2])
            flows_flipped = tf.reverse(flows, axis=[2])
            saliency_flipped = tf.reverse(saliency, axis=[2])
            bboxes_flipped = tf.map_fn(self._flip_single_example, elems=bboxes, parallel_iterations=1)

            return frames_flipped, flows_flipped, saliency_flipped, bboxes_flipped

        flipped = tf.cond(tf.cast(do_hflip, dtype=tf.bool), _hflip, _no_flip)

        # unpack
        frames_flipped, flows_flipped, saliency_flipped, bboxes_flipped = flipped

        return frames_flipped, flows_flipped, saliency_flipped, bboxes_flipped, num_seqs

    # create (features, label) for training
    def _fmt_input(self, frames, flows, saliency, bboxes, num_seqs):
        # computes speed and direction from opticflow
        flow_x = flows[:, :, :, 0]
        flow_y = flows[:, :, :, 1]

        # parse optic flow into speed and direction
        speed = tf.norm(flows, ord=2, axis=3)
        direction = tf.atan2(flow_y, flow_x)

        speed_tents = self._project_speed_tents(speed)

        return {
            'frames': frames,
            'speed': speed,
            'speed_tents': speed_tents,
            'saliency': saliency,
            'direction': direction,
            'bbox': bboxes[0, :],
            'num_seqs_in_video': num_seqs
        }, bboxes[-1, :]

    ##### helper functions used in map setps #####

    # re-order y's and x's in one bbox
    def _reorder_coords_single_bbox(self, bbox):
        y_min = tf.reduce_min(tf.stack([bbox[0], bbox[2]]))
        y_max = tf.reduce_max(tf.stack([bbox[0], bbox[2]]))
        x_min = tf.reduce_min(tf.stack([bbox[1], bbox[3]]))
        x_max = tf.reduce_max(tf.stack([bbox[1], bbox[3]]))

        return tf.stack([y_min, x_min, y_max, x_max])

    # flip the x-axis for one bbox
    def _flip_single_example(self, bbox):
        y_min, x_min, y_max, x_max = tf.unstack(bbox)

        x_min_prime = 1 - x_max
        x_max_prime = 1 - x_min

        return tf.stack([y_min, x_min_prime, y_max, x_max_prime])

    # project speed with tent basis
    def _project_speed_tents(self, speed_seq):
        tent_centers = np.exp(np.arange(0, 5, .45))

        def _project_single_speed_example(speed_scaled):
            tent_basis = []
            for j in range(0, len(tent_centers) - 2):
                _left = tent_centers[j]
                _center = tent_centers[j + 1]
                _right = tent_centers[j + 2]

                y_left = (speed_scaled - _left) / (_center - _left)
                y_right = (_right - speed_scaled) / (_right - _center)

                y = tf.where((speed_scaled >= _left) & (speed_scaled <= _center), y_left, tf.zeros_like(speed_scaled)) \
                    + tf.where((speed_scaled >= _center) & (speed_scaled <= _right), y_right, tf.zeros_like(speed_scaled))

                tent_basis.append(y)

            speed_tents = tf.stack(tent_basis, axis=2)

            return speed_tents

        speed_tents_seq = tf.map_fn(_project_single_speed_example, elems=speed_seq * self.speed_scalar, parallel_iterations=1)

        return speed_tents_seq