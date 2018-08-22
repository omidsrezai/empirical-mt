import sys
sys.path.append('../../')

from datetime import datetime
from time import time
import argparse
from os import path

import tensorflow as tf
import numpy as np

from visual_tracking.model.mt_mst_sequence_tracker import MTMSTSeqTracker
from visual_tracking.data_pipline.speed_direction_saliency_input import SpeedDirectionSaliencySeqInputFunc

SPEED_SCALER = 4

tf.logging.set_verbosity(tf.logging.DEBUG)

best_metric = 0

KERNEL_PATH = 'mt_over_time/time_map_area_mt/area_mt/%s/chann_sel_conv2d/kernel'


def _save_best_kernels(model, model_name):
    tf.logging.info('saving best kernels...')

    to_save = {}

    for area_name in ['excitatory', 'dir_sel_sup', 'non_dir_sel_sup']:
        kernels = model.get_variable_value(KERNEL_PATH % area_name)
        to_save[area_name] = kernels

    np.savez('../weights/%s_weights' % path.basename(model_name), **to_save)
    return


def main(argv=None):
    # training arguments
    parser = argparse.ArgumentParser(description='train visual tracking model')
    parser.add_argument('--train-batch-size', action='store', type=int, default=64)
    parser.add_argument('--test-batch-size', action='store', type=int, default=32)
    parser.add_argument('--learning-rate', action='store', type=float, default=0.0001)
    parser.add_argument('--lr-decay-steps', action='store', type=int, default=100)
    parser.add_argument('--loss', action='store', type=str, default='l1')
    parser.add_argument('--max-im-outputs', action='store', type=int, default=16)
    parser.add_argument('--tensorboard-dir', action='store', type=str, required=True)
    parser.add_argument('--desc', action='store', type=str, required=True)
    parser.add_argument('--short-desc', action='store', type=str, required=True)
    parser.add_argument('--train-shuffle-buffer-size', action='store', type=int, default=100)
    parser.add_argument('--model-name', action='store', type=str, default=None)

    # dataset arguments
    parser.add_argument('--fixed-input-dim', action='store', type=int, default=76)
    parser.add_argument('--k', action='store', type=int, default=2)
    parser.add_argument('--flow-method', action='store', type=str, default='fb')
    parser.add_argument('--train-dataset-cache-id', action='store', type=str, default=None)
    parser.add_argument('--test-dataset-cache-id', action='store', type=str, default=None)
    parser.add_argument('--saliency-method', action='store', type=str, default='mb')
    parser.add_argument('--saliency-folderpath', action='store', type=str, default=None)

    args = parser.parse_args(args=argv[1:])
    tf.logging.debug('args: %s' % args)

    if args.saliency_folderpath is not None:
        saliency_configs = {'saliencymaps_folderpath': args.saliency_folderpath}
    else:
        saliency_configs = {'saliency_method': args.saliency_method}

    input_fn_train = SpeedDirectionSaliencySeqInputFunc(dataset_index_filepath='../../data/alov300++/alov300_train.csv',
                                                        input_folderpath='../../',
                                                        fixed_input_dim=args.fixed_input_dim,
                                                        batch_size=args.train_batch_size,
                                                        num_epochs=1,
                                                        cache_id=args.train_dataset_cache_id,
                                                        flow_method=args.flow_method,
                                                        shuffle=True,
                                                        shuffle_buffer_size=args.train_shuffle_buffer_size,
                                                        k=args.k,
                                                        data_augmentation=True,
                                                        **saliency_configs)

    input_fn_eval = SpeedDirectionSaliencySeqInputFunc(dataset_index_filepath='../../data/alov300++/alov300_test.csv',
                                                       input_folderpath='../../',
                                                       fixed_input_dim=args.fixed_input_dim,
                                                       batch_size=args.test_batch_size,
                                                       num_epochs=1,
                                                       cache_id=args.test_dataset_cache_id,
                                                       flow_method=args.flow_method,
                                                       shuffle=False,
                                                       k=args.k,
                                                       data_augmentation=False,
                                                       **saliency_configs)

    # whether the current session continues a previous session
    cont_train_flag = False

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    if args.model_name is not None:
        cont_train_flag = True
        model_name = args.model_name
    else:
        model_name = '%s_%s' % (args.short_desc, timestamp)
    model_dir = '../../%s/alov300/%s' % (args.tensorboard_dir, model_name)

    model_fn = MTMSTSeqTracker(mt_params_path='../../params_MT_654.pkl',
                               mt_attention_gain_path='../../attention_gains.npy',
                               speed_scalar=SPEED_SCALER,
                               max_image_outputs=args.max_im_outputs,
                               lr=args.learning_rate,
                               lr_decay_steps=args.lr_decay_steps,
                               loss=args.loss)

    m = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    epochs = 0
    best_metric = 0

    while True:
        m.train(input_fn=input_fn_train)
        metrics = m.evaluate(input_fn=input_fn_eval)
        tf.logging.info(metrics)

        if metrics['mean_iou'] > best_metric:
            best_metric = best_metric['mean_iou']
            _save_best_kernels(m, model_name)

        epochs += 1

        if epochs == 5 and not cont_train_flag:
            # log training metadata to a file
            with open("logs/logs.txt", "a") as log:
                log.write("%s: %s\n\n" % (model_dir, args))

if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)