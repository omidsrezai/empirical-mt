import sys
sys.path.append('../../')

import argparse

import tensorflow as tf

from visual_tracking.data_pipline.speed_direction_saliency_input import SpeedDirectionSaliencySeqInputFunc


tf.logging.set_verbosity(tf.logging.DEBUG)


def _cache(next_element):
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        while True:
            try:
                sess.run(next_element)
            except tf.errors.OutOfRangeError:
                return


def main(argv=None):
    # training arguments
    parser = argparse.ArgumentParser(description='train visual tracking model')

    # dataset arguments
    parser.add_argument('--fixed-input-dim', action='store', type=int, default=76)
    parser.add_argument('--k', action='store', type=int, default=2)
    parser.add_argument('--flow-method', action='store', type=str, default='fb')
    parser.add_argument('--train-dataset-cache-id', action='store', type=str, default=None)
    parser.add_argument('--test-dataset-cache-id', action='store', type=str, default=None)
    parser.add_argument('--saliency-method', action='store', type=str, default='mb')
    parser.add_argument('--saliency-folderpath', action='store', type=str, default=None)
    parser.add_argument('--n-workers', action='store', type=int, default=1)
    parser.add_argument('--batch-size', action='store', type=int, default=32)

    args = parser.parse_args(args=argv[1:])
    tf.logging.debug('args: %s' % args)

    if args.saliency_folderpath is not None:
        saliency_configs = {'saliencymaps_folderpath': args.saliency_folderpath}
    else:
        saliency_configs = {'saliency_method': args.saliency_method}

    input_fn_train = SpeedDirectionSaliencySeqInputFunc(dataset_index_filepath='../../data/alov300++/alov300_train.csv',
                                                        input_folderpath='../../',
                                                        fixed_input_dim=args.fixed_input_dim,
                                                        batch_size=args.batch_size,
                                                        num_epochs=1,
                                                        cache_id=args.train_dataset_cache_id,
                                                        flow_method=args.flow_method,
                                                        shuffle=False,
                                                        k=args.k,
                                                        data_augmentation=False,
                                                        n_workers=args.n_workers,
                                                        **saliency_configs)

    input_fn_eval = SpeedDirectionSaliencySeqInputFunc(dataset_index_filepath='../../data/alov300++/alov300_test.csv',
                                                       input_folderpath='../../',
                                                       fixed_input_dim=args.fixed_input_dim,
                                                       batch_size=args.batch_size,
                                                       num_epochs=1,
                                                       cache_id=args.test_dataset_cache_id,
                                                       flow_method=args.flow_method,
                                                       shuffle=False,
                                                       k=args.k,
                                                       data_augmentation=False,
                                                       n_workers=args.n_workers,
                                                       **saliency_configs)

    _cache(input_fn_train())
    _cache(input_fn_eval())


if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)