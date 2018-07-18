import sys
sys.path.append('../../')

from datetime import datetime
from time import time
import argparse

import tensorflow as tf

from visual_tracking.model.mt_mst_sequence_tracker import MTMSTSeqTracker
from visual_tracking.data_pipline.speed_direction_seq_input import SpeedDirectionSeqInputFunc

SPEED_SCALER = 4

tf.logging.set_verbosity(tf.logging.DEBUG)


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

    # dataset arguments
    parser.add_argument('--fixed-input-dim', action='store', type=int, default=76)
    parser.add_argument('--flow-method', action='store', type=str, default='fb')
    parser.add_argument('--train-dataset-cache-id', action='store', type=str, default='train_seq_shuffled_fixed')
    parser.add_argument('--test-dataset-cache-id', action='store', type=str, default='test_seq_shuffled_fixed')

    args = parser.parse_args(args=argv[1:])
    tf.logging.debug('args: %s' % args)

    input_fn_train = SpeedDirectionSeqInputFunc(dataset_index_filepath='../../data/alov300++/alov300_train.csv',
                                                input_path='../../',
                                                fixed_input_dim=args.fixed_input_dim,
                                                batch_size=args.train_batch_size,
                                                num_epochs=1,
                                                cache_id=args.train_dataset_cache_id,
                                                flow_method=args.flow_method,
                                                shuffle=True)

    input_fn_eval = SpeedDirectionSeqInputFunc(dataset_index_filepath='../../data/alov300++/alov300_test.csv',
                                               input_path='../../',
                                               fixed_input_dim=args.fixed_input_dim,
                                               batch_size=args.test_batch_size,
                                               num_epochs=1,
                                               cache_id=args.test_dataset_cache_id,
                                               flow_method=args.flow_method,
                                               shuffle=False)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_dir = '../../%s/alov300/%s_%s' % (args.tensorboard_dir, args.short_desc, timestamp)

    model_fn = MTMSTSeqTracker(mt_params_path='../../params_MT_654.pkl',
                               speed_scalar=SPEED_SCALER,
                               max_image_outputs=args.max_im_outputs,
                               lr=args.learning_rate,
                               lr_decay_steps=args.lr_decay_steps,
                               loss=args.loss)

    m = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    epochs = 0

    while True:
        m.train(input_fn=input_fn_train)
        metrics = m.evaluate(input_fn=input_fn_eval)
        tf.logging.info(metrics)

        epochs += 1

        if epochs == 5:
            # log training metadata to a file
            with open("logs/logs.txt", "a") as log:
                log.write("%s: %s\n\n" % (model_dir, args))

if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)