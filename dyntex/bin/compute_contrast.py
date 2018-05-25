import os

import pandas as pd
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from pelic.calculate_contrast import CalculateContrast
from pelic.gabors import makeGabors, makeGaussian

VIDEO_FILEPATH_PREFIX = '../../input/DynTex++'
CONTRAST_FILEPATH_PREFIX = '../../data/dyntex++/contrast'


if __name__ == '__main__':
    dyntex_df = pd.read_csv('../../data/dyntex++/dyntex.csv')

    # TODO double check with Omid on the parameters
    size_ = 101
    gabors = makeGabors(size_, center=None)
    gaussian = makeGaussian(size_, sigma=5, center=None)

    contrast_calculator = CalculateContrast(gabors, gaussian)

    for video_filepath in tqdm(dyntex_df['filepath']):
        video_filepath = os.path.join(VIDEO_FILEPATH_PREFIX, video_filepath)
        video_filename = os.path.splitext(os.path.basename(video_filepath))[0]
        contrast_filepath = os.path.join(CONTRAST_FILEPATH_PREFIX, video_filename)

        video = sio.loadmat(video_filepath)['subv']
        contrast = contrast_calculator.calculate_contrast(video, is_smoothed=False)

        np.save(contrast_filepath, contrast)


