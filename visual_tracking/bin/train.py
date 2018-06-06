import tensorflow as tf

from visual_tracking.model.alov300_input_pipeline import PairwiseVideoFrameInputFunc
from visual_tracking.model.alov300_main import nn_model_fn

DATASET_DIR = '../../data/alov300++/pairwise_frac0.01_size78_ratio0.3_grayscale'

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    alov300_pairwise_input_fn = PairwiseVideoFrameInputFunc(dataset_dir=DATASET_DIR)
    model_name = tf.estimator.Estimator(nn_model_fn, model_dir='../../serialized_models/alov300')

    tensors_to_log = {"speed": 'mt_inputs/speed_input_tensor',
                      'direction': 'mt_inputs/direction_input_tensor',
                      "MT_feats": "flattened_MT_act_tensor",
                      "predictions": "bbox_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    model_name.train(input_fn=alov300_pairwise_input_fn,
                     steps=1,

                     hooks=[logging_hook])


if __name__ == '__main__':
    tf.app.run(main=main)


