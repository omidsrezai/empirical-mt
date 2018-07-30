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
                 shuffle_buffer_size=200,
                 sequence_length=6,
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
        self.sequence_length = sequence_length
        self.kv_pairs = kv_pairs

    # group frames in a video to sequences of 6 images
    def _group_frames_6_pyfunc(self, video_folderpath, annotation_filepath, height, width):
        # change to absolute path
        video_folderpath_abs = path.join(self.input_path, video_folderpath)
        annotation_filepath_abs = path.join(self.input_path, annotation_filepath)

        gp_frame_ids = []  # shape = (number of sequences, 6, frame_id)
        # shape = (number of sequences, 6, 4)
        gp_bboxes = np.array([], dtype=np.float32).reshape((0, self.sequence_length, 4))
        # shape = (number of sequences,)
        num_sequences = 0  # number of sequences of frames

        # parse one line in annotation files
        def _unpack_annotation_row(r):
            id = int(r[0])

            y_min = float(r[4])
            x_min = float(r[3])
            y_max = float(r[8])
            x_max = float(r[7])

            x_min, x_max = sorted((x_min, x_max))
            y_min, y_max = sorted((y_min, y_max))

            return id, y_min, x_min, y_max, x_max

        if not self.sequence_length == 6:
            tf.logging.warn("%s sequence length not supported at the moment, setting to 6")
            self.sequence_length = 6

        with open(annotation_filepath_abs, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            prev_frame_id = None

            # read the lines in the annotation, produces a dataset of
            #  (consecutive 6 frames, bounding boxes for the 6 frames, number of sequences for a video)
            #  missing annotations are interpolated
            for row in reader:
                if prev_frame_id is None:
                    prev_frame_id, prev_y_min, prev_x_min, prev_y_max, prev_x_max = _unpack_annotation_row(row)

                else:
                    # both frame_id and prev_frame_id have annotations
                    frame_id, y_min, x_min, y_max, x_max = _unpack_annotation_row(row)
                    num_frames = frame_id - prev_frame_id + 1

                    if num_frames < self.sequence_length: # check whether it has max_seq_len frames
                        tf.logging.warn('%s id %s to id %s are annotated too frequently, continue reading more frames',
                                        annotation_filepath_abs, prev_frame_id, frame_id)

                        # keep reading more frames
                        continue

                    # interpolate bounding boxes for the frames between prev_frame_id and frame_id
                    # if num_frames > sequence_length, then sequence_length frames are sampled
                    num_frames = min(num_frames, self.sequence_length)

                    _frame_ids = np.linspace(prev_frame_id, frame_id, num=num_frames, dtype=np.int32).tolist()
                    y_mins = np.linspace(prev_y_min, y_min, num=num_frames, dtype=np.float32) / (height - 1)
                    x_mins = np.linspace(prev_x_min, x_min, num=num_frames, dtype=np.float32) / (width - 1)
                    y_maxs = np.linspace(prev_y_max, y_max, num=num_frames, dtype=np.float32) / (height - 1)
                    x_maxs = np.linspace(prev_x_max, x_max, num=num_frames, dtype=np.float32) / (width - 1)

                    # append results
                    gp_frame_ids.append(_frame_ids[0:num_frames])

                    _bboxes = np.transpose(np.stack([y_mins, x_mins, y_maxs, x_maxs], axis=0))
                    gp_bboxes = np.concatenate([gp_bboxes, np.expand_dims(_bboxes, axis=0)], axis=0)

                    num_sequences += 1

                    # update previous frame
                    prev_frame_id, prev_y_min, prev_x_min, prev_y_max, prev_x_max = (frame_id, y_min, x_min, y_max, x_max)

        # map frame id to frame file path
        if len(gp_frame_ids) > 0:
            gp_frames_filepaths = np.array([[path.join(video_folderpath_abs, '%08d.jpg' % id) for id in ids]
                                            for ids in gp_frame_ids], dtype=np.str)
        else:
            tf.logging.warn("%s has 0 sequence of 6 frames", video_folderpath_abs)
            gp_frames_filepaths = np.array([], dtype=np.str).reshape((0, self.sequence_length))

        return gp_frames_filepaths, gp_bboxes, np.ones(shape=(gp_bboxes.shape[0],), dtype=np.int32) * num_sequences

    # read frames and additional image data from the disk
    def _read_frames_pyfunc(self, frame_filepaths, bboxes, num_seq_in_video):
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

        return frames, bboxes, num_seq_in_video

    # additional steps to preprocess the dataset, preprocessed data will be cached in disk
    def preprocess_cached(self, dataset):
        return dataset

    # additional steps to produce the input to model, this is computed on the fly in every epoch and thus needs
    # to be fast
    def preprocess_no_cache(self, dataset):
        return dataset

    def __call__(self):
        dataset_df = pd.read_csv(self.dataset_index_filepath)

        video_paths = dataset_df['video_path'].tolist()
        annotation_paths = dataset_df['ground_truth_path'].tolist()
        frame_heights = dataset_df['size_x'].tolist()
        frame_widths = dataset_df['size_y'].tolist()

        # parse the input
        dataset = tf.data.Dataset.from_tensor_slices((video_paths, annotation_paths, frame_heights, frame_widths))

        # group dataset into sequences of 6 frames
        dataset = dataset\
            .flat_map(lambda video_path, annotation_path, h, w:
                                                 tf.data.Dataset.from_tensor_slices(tuple(tf.py_func(
                                                     self._group_frames_6_pyfunc,
                                                     [video_path, annotation_path, h, w],
                                                     [tf.string, tf.float32, tf.int32]))))\
            .shuffle(buffer_size=30000)\
            .map(lambda filepaths, bboxes, num_seqs_in_video:
                                    tuple(tf.py_func(
                                        self._read_frames_pyfunc,
                                        [filepaths, bboxes, num_seqs_in_video],
                                        [tf.float32, tf.float32, tf.int32])),
                 num_parallel_calls=self.n_workers)  # all workers are used since there is an IO-bounded python function

        # apply additional pre-processing
        dataset = self.preprocess_cached(dataset)

        # cache to the disk
        if self.cache_id is not None:
            dataset = dataset.cache(filename=path.join(self.cache_path, ('dataset_cache_%s' % self.cache_id)))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffersize)

        # additional steps and feed to the model
        dataset = self.preprocess_no_cache(dataset)\
            .repeat(self.num_epochs) \
            .batch(self.batch_size) \
            .prefetch(2)

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        print('input pipline initilized, input_shpae=%s, input_dtype=%s' %
              (dataset.output_shapes, dataset.output_types))

        return features