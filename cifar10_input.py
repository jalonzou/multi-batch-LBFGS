from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import tarfile

import tensorflow.python.platform
from six.moves import xrange
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.platform import gfile

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data', 'Path to the CIFAR-10 data directory')

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

class CIFAR10Record(object):
	def __init__(self, height=32, width=32, depth=3, key=None, label=None, uint8image=None):
		self._height = height
		self._width = width
		self._depth = depth
		self._key = key
		self._label = label
		self._uint8image = uint8image

	def has_value(self):
		if self._key != None and self._label != None and self._uint8image != None:
			return True
		else:
			return False

	@property
	def height(self):
		return self._height

	@property
	def width(self):
		return self._width

	@property
	def depth(self):
		return self._depth

	@property
	def key(self):
		return self._key
	@key.setter
	def key(self, value):
		self._key = value

	@property
	def label(self):
		return self._label
	@label.setter
	def label(self, value):
		self._label = value

	@property
	def uint8image(self):
		return self._uint8image
	@uint8image.setter
	def uint8image(self, value):
		self._uint8image = value
	


def read_cifar10(filename_queue):
	cifar_record = CIFAR10Record()

	label_bytes = 1
	image_bytes = cifar_record.height * cifar_record.width * cifar_record.depth
	record_bytes = label_bytes + image_bytes

	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	cifar_record.key, value = reader.read(filename_queue)

	record_data = tf.decode_raw(value, tf.uint8)

	cifar_record.label = tf.cast(tf.slice(record_data, [0], [label_bytes]), tf.int32)

	image_data = tf.reshape(tf.slice(record_data, [label_bytes], [image_bytes]), [cifar_record.depth, cifar_record.height, cifar_record.width])
	cifar_record.uint8image = tf.transpose(image_data, [1, 2, 0])

	if cifar_record.has_value():
		return cifar_record
	else:
		raise AttributeError('record data has not been set successfully')


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
	num_preprocess_threads = 8

	image_batch, label_batch = tf.train.shuffle_batch(
		[image, label],
		batch_size=batch_size,
		num_threads=num_preprocess_threads,
		capacity=min_queue_examples + 3 * batch_size,
		min_after_dequeue=min_queue_examples)

	tf.summary.image('images', image_batch)

	return image_batch, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]

	for filename in filenames:
		if not gfile.Exists(filename):
			raise ValueError('Fail to find file:' + filename)

	filename_queue = tf.train.string_input_producer(filenames)

	input_data = read_cifar10(filename_queue)
	reshaped_image = tf.cast(input_data.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	distorted_image = tf.random_crop(reshaped_image, [height, width, input_data.depth])

	distorted_image = tf.image.random_flip_left_right(distorted_image)

	distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)

	distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

	float_image = tf.image.per_image_standardization(distorted_image)

	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes' % min_queue_examples)

	return _generate_image_and_label_batch(float_image, input_data.label, min_queue_examples, batch_size)


def eval_inputs(is_eval_data, data_dir, batch_size):
	if is_eval_data:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	else:
		filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

	for filename in filenames:
		if not gfile.Exists(filename):
			raise ValueError('Fail to find file: ' + filename)

	filename_queue = tf.train.string_input_producer(filenames)

	input_data = read_cifar10(filename_queue)
	reshaped_image = tf.cast(input_data.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

	float_image = tf.image.per_image_standardization(resized_image)

	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	
	return _generate_image_and_label_batch(float_image, input_data.label, min_queue_examples, batch_size)

def maybe_download_and_extract():
	dest_directory = FLAGS.data_dir
	if not os.path.exists(dest_directory):
		os.mkdir(dest_directory)

	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>>Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)

		print()
		statinfo = os.stat(filepath)
		print('successfully downloaded', filename, statinfo.st_size, 'bytes. ')
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)
"""	
	all_filenams = [os.path.join(dest_directory, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
	all_filenams.extend([os.path.join(dest_directory, 'test_batch.bin')])

	tar = tarfile.open(filepath, 'r:gz')

	for f in all_filenams:
		if not gfile.Exists(f):
			filename = 'cifar-10-batches-bin/' + f.split('/')[-1]
			tar.extract(filename, dest_directory)

	tar.close()"""




