from datetime import datetime
from time import time

import tensorflow as tf

from visual_tracking.model.alov300_model_omid_mt2 import MTTracker
from visual_tracking.model.mt_feature_decoder import MTFeatureDecoder
from visual_tracking.utils.est_contrast_flow_input_pip import EstimatorOpticFlowInputFunc

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    input_fn_train =  EstimatorOpticFlowInputFunc(dataset_index_filepath='../../data/alov300++/alov300_train.csv',
                                                  input_path='../../',
                                                  fixed_input_dim=76,
                                                  batch_size=64,
                                                  num_epochs=1)
    input_fn_eval = EstimatorOpticFlowInputFunc(dataset_index_filepath='../../data/alov300++/alov300_test.csv',
                                                  input_path='../../',
                                                  fixed_input_dim=76,
                                                  batch_size=64,
                                                  num_epochs=1)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_dir = '../../models_tmp2/alov300/%s' % timestamp

    model_fn = MTTracker(mt_params_path='../../params_MT_654.pkl', speed_scaler=0.4)

    m = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    while True:
        m.train(input_fn=input_fn_train,
                         steps=None,
                         hooks=[logging_hook])
        m.evaluate(input_fn=input_fn_eval)


if __name__ == '__main__':
    tf.app.run(main=main)