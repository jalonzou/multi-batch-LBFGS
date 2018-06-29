from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange
import tensorflow as tf

import cifar10
import customized_optimizer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'cifar10_log',
                           'Directory where to write event logs and checkpoint')
tf.app.flags.DEFINE_integer('max_steps', 50, 'Number of batches to run')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement')


def placeholder_inputs(batch_size, image_size):
    with tf.device('/cpu:0'):
        images_placeholder = tf.placeholder(tf.float32, shape=(
            batch_size, image_size, image_size, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return images_placeholder, labels_placeholder


def step_callback(step, step_length, duration, loss_val, summary_func, saver, session, eps, spb):
    # num_examples_per_step = FLAGS.batch_size
    # examples_per_sec = num_examples_per_step / duration
    # sec_per_batch = float(duration)

    # if step > 2:
    #     eps.append(examples_per_sec)
    #     spb.append(sec_per_batch)

    if step % 1 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = (
            '%s: step %d, loss = %.3f, duration:%.3fs, next_step_length = %.3f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), step, loss_val, duration,
                            step_length, examples_per_sec, sec_per_batch))

    if step % 100 == 0:
        summary_func(step)

    if step % 50 == 0 or (step + 1) == FLAGS.max_steps:
        # if not step == 0:
        #     print('eps:%.3f' % (sum(eps)/len(eps)))
        #     print('spb:%.3f' % (sum(spb)/len(spb)))
        checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(session, checkpoint_path, global_step=step)


def run_training():
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, trainable=False)

        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size, cifar10.IMAGE_SIZE)

        logits = cifar10.inference(images_placeholder)
        losses_dict = cifar10.loss(logits, labels_placeholder)

        moving_averages_op = cifar10.add_summaries_and_moving_avgs(losses_dict, global_step)

        lbfgs_optimizer = customized_optimizer.CustomizedOptimizerInterface(
            global_step=global_step,
            loss_dict=losses_dict,
            data_fetches=[images, labels],
            data_placeholders=(images_placeholder, labels_placeholder),
            maxiter=FLAGS.max_steps)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=25)

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4))) as sess:
            sess.run(init)

            coordinator = tf.train.Coordinator()
            try:
                threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

                lbfgs_optimizer.minimize(session=sess,
                                         moving_averages_op=moving_averages_op,
                                         summary_op=summary_op,
                                         saver=saver,
                                         step_callback=step_callback)

            except Exception as e:
                coordinator.request_stop(e)

            coordinator.request_stop()
            coordinator.join(threads, stop_grace_period_secs=10)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if gfile.Exists(FLAGS.log_dir):
        gfile.DeleteRecursively(FLAGS.log_dir)
    gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    tf.app.run()
