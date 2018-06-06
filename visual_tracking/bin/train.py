import tensorflow as tf

from visual_tracking.model.alov300_input_pipeline import PairwiseVideoFrameInputFunc
from visual_tracking.model.alov300_main import nn_model_fn

DATASET_DIR = '../../data/alov300++/pairwise_frac0.01_size78_ratio0.3_grayscale'

if __name__ == '__main__':
    alov300_pairwise_input_fn = PairwiseVideoFrameInputFunc(dataset_dir=DATASET_DIR)
    model_name = tf.estimator.Estimator(nn_model_fn, model_dir= '../../serialized_models/alov300')

    model_name.train(input_fn=alov300_pairwise_input_fn, steps=1)
