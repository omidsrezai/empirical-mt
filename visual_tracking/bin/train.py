from datetime import datetime
from time import time

import tensorflow as tf

from visual_tracking.utils.alov300_input_pipeline import SpeedDirectionInputFunc, OptFlowContrastInputFunc, SearchPatchObjPatchInputFn
from visual_tracking.model.alov300_model_omid_mt import ALOV300OmideMTModelFn
from visual_tracking.model.alov300_model_cnn import ALOV300CNNFn
from visual_tracking.model.alov300_model_goturn import ALOV300GOTURNFn

TRAIN_DATASET_DIR = '../../data/alov300++/pairwise_train_frac1_size76_ratio0.5_grayscale'  # TODO make this an input
EVAL_DATASET_DIR = '../../data/alov300++/pairwise_test_frac1_size76_ratio0.5_grayscale'

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    alov300_pairwise_input_fn_train = SpeedDirectionInputFunc(dataset_dir=TRAIN_DATASET_DIR, num_epochs=1)
    # alov300_pairwise_input_fn_train = OptFlowContrastInputFunc(dataset_dir=TRAIN_DATASET_DIR, num_epochs=1)
    input_fn_eval = SpeedDirectionInputFunc(dataset_dir=EVAL_DATASET_DIR, num_epochs=1)

    # input_fn_eval = OptFlowContrastInputFunc(dataset_dir=EVAL_DATASET_DIR, num_epochs=1)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_dir = '../../models_tmp/alov300/%s' % timestamp

    model_fn = ALOV300OmideMTModelFn(mt_params_path='../../params_MT_654.pkl')
    # model_fn = ALOV300CNNFn()

    model_name = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    tensors_to_log = {} #"speed": 'mt_inputs/speed_input_tensor',}
                      #'direction': 'mt_inputs/direction_input_tensor',
                      #"MT_feats": "tracker_nn/flattened_mt_act_tensor"}
                      #"dense": 'tracker_nn/dense_layer/BiasAdd'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    while True:


        model_name.train(input_fn=alov300_pairwise_input_fn_train,
                         steps=None,
                         hooks=[logging_hook])
        model_name.evaluate(input_fn=input_fn_eval)


if __name__ == '__main__':
    tf.app.run(main=main)