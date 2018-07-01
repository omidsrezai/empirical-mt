from datetime import datetime
from time import time

import tensorflow as tf

from visual_tracking.utils.alov300_input_pipeline import SpeedDirectionInputFunc, OptFlowContrastInputFunc, SearchPatchObjPatchInputFn
from visual_tracking.model.alov300_model_omid_mt import ALOV300OmideMTModelFn
from visual_tracking.model.alov300_model_cnn import ALOV300CNNFn
from visual_tracking.model.alov300_model_goturn import ALOV300GOTURNFn

from os import path

DATASET_DIR = '../../data/alov300++/pairwise_test_frac1_size76_ratio0.5_grayscale'  # TODO make this an input

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    alov300_pairwise_input_fn = SpeedDirectionInputFunc(dataset_dir=DATASET_DIR, num_epochs=1)

    model_dir = path.join('../../models_tmp/alov300', '06-14-17-45-00')

    model_fn = ALOV300OmideMTModelFn(mt_params_path='../../params_MT_654.pkl')

    model_name = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    model_name.evaluate(input_fn=alov300_pairwise_input_fn)



if __name__ == '__main__':
    tf.app.run(main=main)