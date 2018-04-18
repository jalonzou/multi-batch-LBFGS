from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('fetched_batch_size', 200, 'Number of images fetched by reader')
tf.app.flags.DEFINE_float('overlapped_rate', 0.2,
                          'Percentage of examples overlapped between two batches')
tf.app.flags.DEFINE_integer('batch_size', FLAGS.fetched_batch_size +
                            int(FLAGS.overlapped_rate * float(FLAGS.fetched_batch_size)),
                            'Number of images to process')

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))

    if wd:
        weight_dacay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_dacay)
        tf.add_to_collection('left_losses', weight_dacay)
        tf.add_to_collection('right_losses', weight_dacay)
    return var


def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please input a data directory')
    file_data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=file_data_dir, batch_size=FLAGS.fetched_batch_size)


def eval_inputs(is_eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please input a data directory')
    file_data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.eval_inputs(is_eval_data=is_eval_data, data_dir=file_data_dir, batch_size=FLAGS.batch_size)


def maybe_download_and_extract():
    return cifar10_input.maybe_download_and_extract()


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME'), biases)
        conv1 = tf.nn.relu(conv, name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.local_response_normalization(
        pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        conv = tf.nn.bias_add(tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME'), biases)
        conv2 = tf.nn.relu(conv, name=scope.name)

    norm2 = tf.nn.local_response_normalization(
        conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('flc3') as scope:
        dim = 1
        for dim_next in pool2.get_shape()[1:].as_list():
            dim *= dim_next
        reshape_feature_map = tf.reshape(pool2, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        flc3 = tf.nn.relu(tf.matmul(reshape_feature_map, weights) + biases, name=scope.name)

    with tf.variable_scope('flc4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        fcl4 = tf.nn.relu(tf.matmul(flc3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fcl4, weights), biases, name=scope.name)

    return softmax_linear


def loss(logits, labels):
    sparse_labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(range(FLAGS.batch_size), 1)
    concated_labels = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated_labels, [FLAGS.batch_size, NUM_CLASSES], 1.0, 0.0)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=dense_labels, name='cross_entropy_per_example')

    num_examples_overlapped = FLAGS.batch_size - FLAGS.fetched_batch_size
    cross_entropy_left = cross_entropy[:num_examples_overlapped]
    cross_entropy_right = cross_entropy[-num_examples_overlapped:]

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='mean_value_of_cross_entropy')
    cross_entropy_mean_left = tf.reduce_mean(
        cross_entropy_left, name='mean_value_of_left_cross_entropy')
    cross_entropy_mean_right = tf.reduce_mean(
        cross_entropy_right, name='mean_value_of_right_cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)
    tf.add_to_collection('left_losses', cross_entropy_mean_left)
    tf.add_to_collection('right_losses', cross_entropy_mean_right)

    losses_dict = {
        'left_loss': tf.add_n(tf.get_collection('left_losses'), name='left_loss'),
        'total_loss': tf.add_n(tf.get_collection('losses'), name='total_loss'),
        'right_loss': tf.add_n(tf.get_collection('right_losses'), name='right_loss')
    }

    return losses_dict


def add_summaries_and_moving_avgs(losses_dict, global_step):

    # attach a histogram summary to all trainable variables, 
    # then calculate their moving averages and maintain them in 
    # shadow variables

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    variable_averages_calculator = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages_calculator.apply(tf.trainable_variables())

    # Calculate the moving averages for each loss in losses collection,
    # including total_loss computed in 'loss' op.
    # then attach scalar summary to each loss aforementioned.
    # Name original losses with 'raw' suffix and their moving averages
    # with their original op name

    total_loss = losses_dict['total_loss']

    loss_averages_calculator = tf.train.ExponentialMovingAverage(0.9, name='avg_loss')
    each_loss = tf.get_collection('losses')
    loss_averages_op = loss_averages_calculator.apply(each_loss + [total_loss])

    for loss in each_loss + [total_loss]:
        tf.summary.scalar(loss.op.name + '(raw)', loss)
        tf.summary.scalar(loss.op.name, loss_averages_calculator.average(loss))

    moving_averages_op = tf.group(variable_averages_op, loss_averages_op)

    return moving_averages_op




