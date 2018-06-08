from datetime import datetime
from time import time

import tensorflow as tf

from visual_tracking.model.alov300_input_pipeline import PairwiseVideoFrameInputFunc
from visual_tracking.model.alov300_main import ALOV300ModelFn

DATASET_DIR = '../../data/alov300++/pairwise_train_frac1_size76_ratio0.3_grayscale'  # TODO make this an input

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    alov300_pairwise_input_fn = PairwiseVideoFrameInputFunc(dataset_dir=DATASET_DIR)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_dir = '../../models/alov300/%s' % timestamp

    model_fn = ALOV300ModelFn(mt_params_path='../../params_MT_654.pkl')

    model_name = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    tensors_to_log = {#"speed": 'mt_inputs/speed_input_tensor',
                      #'direction': 'mt_inputs/direction_input_tensor',
                      "MT_feats": "tracker_nn/flattened_mt_act_tensor"}
                      #"dense": 'tracker_nn/dense_layer/BiasAdd'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    model_name.train(input_fn=alov300_pairwise_input_fn,
                     steps=None,
                     hooks=[logging_hook])


if __name__ == '__main__':
    tf.app.run(main=main)