from os import listdir, path
import numpy as np
from skimage import io
from tqdm import tqdm

from pelic.calculate_contrast import CalculateContrast
from pelic.gabors import makeGabors, makeGaussian

FIXED_FRAME_SIZE = 78
DATASET_DIR = '../../data/alov300++/pairwise_size78_ratio0.3_grayscale'

size_ = 101
gabors = makeGabors(size_, center=None)
gaussian = makeGaussian(size_, sigma=5, center=None)
contrast_calculator = CalculateContrast(gabors, gaussian)


def _compute_contrast_in_chunk(video_dir_path):
    # print('processing %s' % video_dir_path)

    frame_folders = listdir(video_dir_path)

    video = np.zeros((FIXED_FRAME_SIZE, FIXED_FRAME_SIZE, len(frame_folders)))

    # print('reading frames...')
    # read frames
    for i, f in enumerate(frame_folders):
        try:
            frame = io.imread(path.join(video_dir_path, f, 'frame.png'))
        except IOError as e:
            print(e)
            frame = np.zeros((FIXED_FRAME_SIZE, FIXED_FRAME_SIZE))

        # checking the frame size
        assert frame.shape == (FIXED_FRAME_SIZE, FIXED_FRAME_SIZE)

        video[:, :, i] = frame

    # compute contrast
    contrast = contrast_calculator.calculate_contrast(video, is_smoothed=False)

    # print('writing contrast...')
    for i, f in enumerate(frame_folders):
        try:
            np.save(path.join(video_dir_path, f, 'contrast'), contrast[:, :, i])
        except IndexError as e:
            print(e, video_dir_path, f, contrast.shape)
            raise e


if __name__ == '__main__':
    video_folders = listdir(DATASET_DIR)
    video_folders = filter(lambda f: path.isdir(path.join(DATASET_DIR, f)), video_folders)

    for f in tqdm(video_folders):
        _compute_contrast_in_chunk(path.join(DATASET_DIR, f))