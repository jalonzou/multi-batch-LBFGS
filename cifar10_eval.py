from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval_log',
                           'Directory where to write event logs.')
tf.app.flags.DEFINE_string('eval_data', 'test',
                           "Either 'test' or 'train'_eval.")
tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_log',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            'How often to run the eval.')
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            'Number of examples to run.')
tf.app.flags.DEFINE_boolean('run_once', True,
                            'Where to run eval only once.')


def eval_once(saver, top_k_op, summary_op):
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph=sess.graph)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for path in ckpt.all_model_checkpoint_paths:
                print(path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(top_k_op)
                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default():
        is_eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.eval_inputs(is_eval_data=is_eval_data)

        logits = cifar10.inference(images)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages_calculator = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages_calculator.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        while True:
            eval_once(saver, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
