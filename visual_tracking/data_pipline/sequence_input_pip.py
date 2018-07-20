import csv
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, img_as_float32, color


class SequenceInputFuncBase(object):

    def __init__(self,
                 dataset_index_filepath,
                 cache_id,
                 input_folderpath='../',
                 cache_folderpath='../../',
                 batch_size=64,
                 num_epochs=-1,
                 shuffle=False,
                 n_workers=10,
                 shuffle_buffer_size=2000,
                 max_seq_len=6,
                 **kv_pairs):
        self.cache_path = cache_folderpath
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset_index_filepath = dataset_index_filepath
        self.input_path = input_folderpath
        self.n_workers = n_workers
        self.cache_id = cache_id
        self.shuffle = shuffle
        self.shuffle_buffersize = shuffle_buffer_size
        self.max_seq_len = max_seq_len
        self.kv_pairs = kv_pairs

    def _group_frames_in_6_pyfunc(self, video_folderpath, annotation_filepath, height, width):
        # change to relative path
        video_folderpath_abs = path.join(self.input_path, video_folderpath)
        annotation_filepath_abs = path.join(self.input_path, annotation_filepath)

        gp_frame_ids = []
        self.max_seq_len = 6
        gp_bboxes = np.array([], dtype=np.float32).reshape((0, self.max_seq_len, 4))

        def _unpack_annotation_row(r):
            id = int(r[0])

            y_min = float(r[4])
            x_min = float(r[3])
            y_max = float(r[8])
            x_max = float(r[7])

            x_min, x_max = sorted((x_min, x_max))
            y_min, y_max = sorted((y_min, y_max))

            return id, y_min, x_min, y_max, x_max

        with open(annotation_filepath_abs, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            prev_frame_id = None

            for row in reader:
                if prev_frame_id is None:
                    prev_frame_id, prev_y_min, prev_x_min, prev_y_max, prev_x_max = _unpack_annotation_row(row)
                else:
                    frame_id, y_min, x_min, y_max, x_max = _unpack_annotation_row(row)
                    num_frames = frame_id - prev_frame_id + 1

                    if num_frames < self.max_seq_len:
                        tf.logging.warn('%s id %s to id %s has non-standard annotations',
                                        annotation_filepath_abs, prev_frame_id, frame_id)
                        prev_frame_id = frame_id
                        continue

                    num_frames = min(num_frames, self.max_seq_len)

                    # interpolate bounding boxes for the frames between annotated frames
                    _frame_ids = np.linspace(prev_frame_id, frame_id, num=num_frames, dtype=np.int32).tolist()
                    y_mins = np.linspace(prev_y_min, y_min, num=num_frames, dtype=np.float32) / (height - 1)
                    x_mins = np.linspace(prev_x_min, x_min, num=num_frames, dtype=np.float32) / (width - 1)
                    y_maxs = np.linspace(prev_y_max, y_max, num=num_frames, dtype=np.float32) / (height - 1)
                    x_maxs = np.linspace(prev_x_max, x_max, num=num_frames, dtype=np.float32) / (width - 1)

                    _bboxes = np.transpose(np.stack([y_mins, x_mins, y_maxs, x_maxs], axis=0))

                    # append results
                    gp_frame_ids.append(_frame_ids[0:num_frames])
                    gp_bboxes = np.concatenate([gp_bboxes, np.expand_dims(_bboxes, axis=0)], axis=0)

                    # update previous frame
                    prev_frame_id, prev_y_min, prev_x_min, prev_y_max, prev_x_max = (frame_id, y_min, x_min, y_max, x_max)

        # map frame id to frame file path
        if len(gp_frame_ids) > 0:
            gp_frames_filepaths = np.array([[path.join(video_folderpath_abs, '%08d.jpg' % id) for id in ids]
                                            for ids in gp_frame_ids], dtype=np.str)
        else:
            gp_frames_filepaths = np.array([], dtype=np.str).reshape((0, self.max_seq_len))

        return gp_frames_filepaths, gp_bboxes

    def _read_frames_pyfunc(self, frame_filepaths, bboxes):
        frames = []

        for fp in frame_filepaths:
            frame = img_as_float32(io.imread(fp))

            # convert gray scale frame to rgb
            if len(frame.shape) == 2:
                frame = color.gray2rgb(frame)
                tf.logging.info('converting %s from grayscale to rgb', fp)

            # load saliency maps and concatenate it to the 4th channel
            if 'saliencymaps_folderpath' in self.kv_pairs:
                saliencymap_path = fp.replace('imagedata++', self.kv_pairs['saliencymaps_folderpath'])
                saliencymap = img_as_float32(io.imread(saliencymap_path))
                frame = np.concatenate([frame, np.expand_dims(saliencymap, axis=2)], axis=2)

            frames.append(frame)

        frames = np.stack(frames, axis=0)

        return frames, bboxes

    def preprocess(self, dataset):
        return dataset

    def format_input(self, dataset):
        return dataset

    def __call__(self):
        dataset_df = pd.read_csv(self.dataset_index_filepath)

        video_paths = dataset_df['video_path'].tolist()
        annotation_paths = dataset_df['ground_truth_path'].tolist()
        frame_heights = dataset_df['size_x'].tolist()
        frame_widths = dataset_df['size_y'].tolist()

        # parse the input
        dataset = tf.data.Dataset.from_tensor_slices((video_paths, annotation_paths, frame_heights, frame_widths))
        print(dataset)

        dataset = dataset\
            .flat_map(lambda video_path, annotation_path, h, w:
                                                 tf.data.Dataset.from_tensor_slices(tuple(tf.py_func(
                                                     self._group_frames_in_6_pyfunc,
                                                     [video_path, annotation_path, h, w],
                                                     [tf.string, tf.float32]))))\
            .shuffle(buffer_size=30000)\
            .map(lambda filepaths, bboxes:
                                    tuple(tf.py_func(
                                        self._read_frames_pyfunc,
                                        [filepaths, bboxes],
                                        [tf.float32, tf.float32])),
                 num_parallel_calls=self.n_workers)

        # apply addtional pre-processing
        dataset = self.preprocess(dataset)

        if self.cache_id is not None:
            dataset = dataset.cache(filename=path.join(self.cache_path, ('dataset_cache_%s' % self.cache_id)))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffersize)

        dataset = self.format_input(dataset)\
            .repeat(self.num_epochs) \
            .batch(self.batch_size) \
            .prefetch(2)

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        print('input pipline initilized, input_shpae=%s, input_dtype=%s' %
              (dataset.output_shapes, dataset.output_types))

        return features