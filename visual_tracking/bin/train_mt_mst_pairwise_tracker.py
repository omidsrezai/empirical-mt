from datetime import datetime
from time import time

import tensorflow as tf

from visual_tracking.model.alov300_model_omid_mt2 import MTTracker
from visual_tracking.model.mt_feature_decoder import MTFeatureDecoder
from visual_tracking.model.mt_mst_visual_tracker import MTMSTPairwiseTracker
from visual_tracking.utils.est_contrast_flow_input_pip import EstimatorOpticFlowInputFunc

SPEED_SCALER = 4

BATCH_SIZE = 64

DESC = 'test'

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args):
    input_fn_train =  EstimatorOpticFlowInputFunc(dataset_index_filepath='../../data/alov300++/alov300_train.csv',
                                                  input_path='../../',
                                                  fixed_input_dim=128,
                                                  batch_size=BATCH_SIZE,
                                                  num_epochs=1,
                                                  cache_id='train_shuffled_with_mask_speed_tents_128',
                                                  flow_method='fb',
                                                  shuffle=True)
    input_fn_eval = EstimatorOpticFlowInputFunc(dataset_index_filepath='../../data/alov300++/alov300_test.csv',
                                                input_path='../../',
                                                fixed_input_dim=128,
                                                batch_size=BATCH_SIZE,
                                                num_epochs=1,
                                                cache_id='eval_with_mask_speed_tents_128',
                                                flow_method='fb',
                                                shuffle=False,
                                                speed_scalar=4)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_dir = '../../models_test_mt_mst_pairwise/alov300/%s_s=%s_%s' % (DESC, SPEED_SCALER, timestamp)
    # model_dir =  '../../models_e2e/alov300/s=4_07-01-17-29-07'

    model_fn = MTMSTPairwiseTracker(mt_params_path='../../params_MT_654.pkl',
                                    speed_scaler=SPEED_SCALER,
                                    max_image_outputs=256,
                                    lr=0.0001)

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