from datetime import datetime
from time import time

import tensorflow as tf

from visual_tracking.model.mt_feature_decoder import MTFeatureDecoder
from visual_tracking.utils.alov300_input_pip_deprecated import SpeedDirectionInputFunc
from visual_tracking.utils.est_contrast_flow_input_pip import EstimatorOpticFlowInputFunc

TRAIN_DATASET_DIR = '../../data/alov300++/pairwise_train_frac1_size76_ratio0.5_grayscale'  # TODO make this an input
EVAL_DATASET_DIR = '../../data/alov300++/pairwise_test_frac1_size76_ratio0.5_grayscale'

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    input_fn_train =  SpeedDirectionInputFunc(dataset_dir=TRAIN_DATASET_DIR, num_epochs=1)
    input_fn_eval = SpeedDirectionInputFunc(dataset_dir=EVAL_DATASET_DIR, num_epochs=1)

    # EstimatorOpticFlowInputFunc(fixed_input_dim=76, dataset_index_filepath='../../data/alov300++/alov300_train.csv', input_path='../../')
    # input_fn_train = OptFlowContrastInputFunc(dataset_dir=TRAIN_DATASET_DIR, num_epochs=1)
    # input_fn_eval = SpeedDirectionInputFunc(dataset_dir=EVAL_DATASET_DIR, num_epochs=1)

    # input_fn_eval = OptFlowContrastInputFunc(dataset_dir=EVAL_DATASET_DIR, num_epochs=1)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_dir = '../../models_tmp2/alov300/%s' % timestamp

    model_fn = MTFeatureDecoder(mt_params_path='../../params_MT_654.pkl')
    # model_fn = ALOV300CNNFn()

    m = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    tensors_to_log = {} #"speed": 'mt_inputs/speed_input_tensor',}
                      #'direction': 'mt_inputs/direction_input_tensor',
                      #"MT_feats": "tracker_nn/flattened_mt_act_tensor"}
                      #"dense": 'tracker_nn/dense_layer/BiasAdd'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    # while True:

    while True:
        m.train(input_fn=input_fn_train,
                         steps=None,
                         hooks=[logging_hook])
        m.evaluate(input_fn=input_fn_eval)


if __name__ == '__main__':
    tf.app.run(main=main)