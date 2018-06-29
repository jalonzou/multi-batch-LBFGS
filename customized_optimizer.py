from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

import minimize_lbfgs

__all__ = ['ExternalOptimizerInterface', 'CustomizedOptimizerInterface']


FLAGS = tf.app.flags.FLAGS


class ExternalOptimizerInterface(object):

    def __init__(self, global_step,
                 loss_dict,
                 data_fetches,
                 data_placeholders,
                 var_list=None,
                 **optimizer_kwargs):

        self._global_step = global_step
        self._cur_step = 0
        self._global_step_update = None

        self._left_loss = loss_dict['left_loss']
        self._right_loss = loss_dict['right_loss']
        self._loss = loss_dict['total_loss']

        if var_list is None:
            self._vars = variables.trainable_variables()
        else:
            self._vars = list(var_list)

        self._update_placeholders = [
            array_ops.placeholder(var.dtype) for var in self._vars]
        self._var_updates = [var.assign(array_ops.reshape(placeholder, _get_shape_tuple(var)))
                             for var, placeholder in zip(self._vars, self._update_placeholders)]

        loss_grads = _compute_gradients(self._loss, self._vars)
        left_loss_grads = _compute_gradients(self._left_loss, self._vars)
        right_loss_grads = _compute_gradients(self._right_loss, self._vars)

        self.optimizer_kwargs = optimizer_kwargs

        self._packed_var = self._pack(self._vars)
        self._packed_loss_grad = self._pack(loss_grads)
        self._packed_left_loss_grad = self._pack(left_loss_grads)
        self._packed_right_loss_grad = self._pack(right_loss_grads)

        # construct slice object using to trace the indices of elements
        # of x and gradient in a flattened array
        dims = [_prod(_get_shape_tuple(var)) for var in self._vars]
        accumulated_dims = list(_accumulate(dims))
        self._packing_slices = [slice(start, end) for start, end in zip(
            accumulated_dims[:-1], accumulated_dims[1:])]

        self.data_fetches = data_fetches
        self.images_placeholder = data_placeholders[0]
        self.labels_placeholder = data_placeholders[1]
        self.fetched_images_list = []
        self.fetched_labels_list = []
        self.data_images = None
        self.data_labels = None

    def minimize(self,
                 session=None,
                 feed_dict=None,
                 fetches=None,
                 step_callback=None,
                 loss_callback=None,
                 **run_kwargs):
        """This is the interface of this optimizer which can be invoked
        using Optimizer.minimize(). The optimizer manages session by
        minimizing trainable variables in the graph, and traces necessary
        info by calling 'callback' funcs periodically.

        Note that you can pass extra tensorflow Saver object to record
        state information every certain steps, and summary_op argument to 
        run this op in 'callback' funcs.

        Args:
          session: A 'Session' instance.
          feed_dict: A feed dict to be passed to calls to 'session.run'.
          fetches: A list of tensors to fetch and supply to 'loss_callback'
            as positional arguments.
          step_callback: A function to be called at each optimization step;
            this func will be revised by adding extra arguments before passing
            it to actual minimizing func.
          loss_callback: A function to be called every time the loss and gradients
            are computed, with evaluated fetches supplied as positional arguments.
          **run_kwargs: kwargs including objects to record states of session.

        Returns:
          This interface has no returns.

        Raises:
          NotImplementedError: If _minimize func is not implemented 
            in its subclass.
        """
        session = session or ops.get_default_session()
        feed_dict = feed_dict or {}
        fetches = fetches or []

        loss_callback = loss_callback or (lambda *fetches: None)
        step_callback = step_callback or (lambda xk: None)

        loss_grad_ops = [self._loss,
                         self._packed_left_loss_grad,
                         self._packed_loss_grad,
                         self._packed_right_loss_grad]

        if 'moving_averages_op' in run_kwargs.keys():
            loss_grad_ops.insert(0, run_kwargs['moving_averages_op'])

        loss_grad_func = self._make_eval_func(
            loss_grad_ops, session, feed_dict, fetches, loss_callback)

        single_loss_func = self._make_single_func(self._loss, session)

        single_grad_func = self._make_single_func(self._packed_loss_grad, session)

        initial_packed_var_val = session.run(self._packed_var)

        if 'summary_op' in run_kwargs.keys() and 'saver' in run_kwargs.keys():
            summary_func = self._make_summary_func(run_kwargs['summary_op'], session)
            revised_step_callback = self._make_callback(session,
                                                        summary_func,
                                                        run_kwargs['saver'],
                                                        step_callback)
        else:
            revised_step_callback = None

        packed_var_val = self._minimize(
            initial_val=initial_packed_var_val,
            loss_grad_func=loss_grad_func,
            single_loss_func=single_loss_func,
            single_grad_func=single_grad_func,
            step_callback=revised_step_callback,
            optimizer_kwargs=self.optimizer_kwargs)
        return packed_var_val
        

    def _minimize(self,
                  initial_val,
                  loss_grad_func,
                  single_loss_func,
                  single_grad_func,
                  step_callback,
                  optimizer_kwargs):
        raise NotImplementedError(
            'To use ExternalOptimizerInterface, subclass from it and implement '
            'the _minimize() method.')

    @classmethod
    def _pack(cls, tensors):
        if not tensors:
            return None
        elif len(tensors) == 1:
            return array_ops.reshape(tensors[0], [-1])
        else:
            flattened = [array_ops.reshape(tensor, [-1]) for tensor in tensors]
            return array_ops.concat(flattened, 0)

    def _make_eval_func(self, tensors, session, feed_dict, fetches, callback=None):
        """Construct loss_and_grad func.

        The func constructed takes a flattend variables x as its argument and then
        update all trainable variables in the graph.

        Before running updating process, a new data batch is built and older batch
        in this class will be replaced, therefore we are able to run evaluating
        process with next training point.

        After variables are updated, we run session to fetch values in tensors
        argument. Those in fetches argument will be added to this process at the
        same time and corresponding results will be returned by calling 'callback'.

        Args:
          tensors: A list of 'tensors' to fetch in session.run.
          session: A 'Session' instance.
          feed_dict: A feed dict to be passed to calls to 'session.run'.
          fetches: A list of tensors to fetch and supply to 'loss_callback'
            as positional arguments.
          callback: A function to be called every time the loss and gradients
            are computed, with evaluated fetches supplied as positional arguments.

        Returns:
          A evaluation function updating variables in the graph and evaluating
          values in tensors and fetches arguments.

        Raises:
          AttributeError: if _make_data_batch func failed to fill data.
        """
        if not isinstance(tensors, list):
            tensors = [tensors]
        num_tensors = len(tensors)

        def eval_func(x):
            self._make_data_batch(session)

            var_vals = [x[packing_slice]
                        for packing_slice in self._packing_slices]

            vars_feed_dict = dict(zip(self._update_placeholders, var_vals))
            session.run(self._var_updates, feed_dict=vars_feed_dict)

            self._cur_step += 1
            session.run(self._global_step.assign(self._cur_step))

            if self.data_images is None or self.data_labels is None:
                raise AttributeError('data batch has not been constructed yet')

            data_feed_dict = {
                self.images_placeholder: self.data_images,
                self.labels_placeholder: self.data_labels
            }

            data_feed_dict.update(feed_dict)

            augmented_fetches = tensors + fetches

            augmented_fetch_vals = session.run(
                augmented_fetches, feed_dict=data_feed_dict)

            if callable(callback):
                callback(*augmented_fetch_vals[num_tensors:])

            return augmented_fetch_vals[:num_tensors]

        return eval_func

    def _make_eval_funcs(self, tensors, session, feed_dict, fetches, callback=None):
        return [
            self._make_eval_func(tensor, session, feed_dict, fetches, callback)
            for tensor in tensors
        ]

    def _make_single_func(self, tensor, session):
        """Construct single evaluating func.

        The func returned is similar to those returned by _make_eval_func.
        The main difference is that this func only replace variables in the
        graph temporarily and only evaluate one value each time. After deriving
        the result, variables are restored to its original values.

        Funcs constructed by this method are mainly invoked during step 
        finding process.

        Args:
          tensor: A tensor to fetch in session.run.
          session: A 'Session' instance.

        Returns:
          A single-tensor evaluation function fetching required value passed
          in the argument 'tensor'.

        Raises:
            AttributeError: if there is no data in data_images and data_labels.
        """
        def single_func(x):
            augmented_feed_dict = {
                var: x[packing_slice].reshape(_get_shape_tuple(var))
                for var, packing_slice in zip(self._vars, self._packing_slices)
            }

            if self.data_images is None or self.data_labels is None:
                raise AttributeError('data batch has not been constructed yet')

            data_feed_dict = {
                self.images_placeholder: self.data_images,
                self.labels_placeholder: self.data_labels
            }

            augmented_feed_dict.update(data_feed_dict)

            fetch_val = session.run(tensor, feed_dict=augmented_feed_dict)

            return fetch_val

        return single_func

    def _make_summary_func(self, tensor, session):
        """Construct summary func.

        Fill placeholders in graph with training data and run session to
        obtain serialized summary string.

        Invoke func returned while calling step_callback func.

        Args:
          tensor: A summary_op to run in session.
          session: A 'Session' instance.

        Returns:
          A summary func which is able to run out serialized summary data
          and wirte it to disk by a summary writer instance.
        """
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=session.graph)

        def summary_func(step):
            data_feed_dict = {
                self.images_placeholder: self.data_images,
                self.labels_placeholder: self.data_labels
            }

            summary_str = session.run(tensor, feed_dict=data_feed_dict)
            summary_writer.add_summary(summary_str, step)

        return summary_func

    def _make_data_batch(self, session):
        """Build a new batch of training data for next evaluation step.

        This func runs session at first to fetch a new batch of data in the
        pipeline. Then we assembled the last two batchs fetched in 'fetched_list'
        to be an 'augmented' data batch with overlapping data points on the left
        and right sides.

        Specification of this method is illustrated below:
                111111111111111
                |              222222222222222
                |--------------|---|          333333333333333
                  assembled bat|ch 1              |
                               |------------------|
                                 assembled batch 2
        Args:
          session: A 'Session' instance.

        Returns:
          This func has no returns except for filling data_images and data_labels
          in class with new training data batches.
        """
        if self.data_images is None:
            for i in xrange(2):
                fetched_images, fetched_labels = session.run(self.data_fetches)
                self.fetched_images_list.append(fetched_images)
                self.fetched_labels_list.append(fetched_labels)

        else:
            fetched_images, fetched_labels = session.run(self.data_fetches)
            self.fetched_images_list.append(fetched_images)
            self.fetched_labels_list.append(fetched_labels)

            del self.fetched_images_list[0]
            del self.fetched_labels_list[0]

        num_examples_overlapped = FLAGS.batch_size - FLAGS.fetched_batch_size

        added_images = self.fetched_images_list[-1][:num_examples_overlapped]
        self.data_images = np.concatenate((self.fetched_images_list[-2], added_images), axis=0)

        added_labels = self.fetched_labels_list[-1][:num_examples_overlapped]
        self.data_labels = np.concatenate((self.fetched_labels_list[-2], added_labels), axis=0)

    def _make_callback(self, session, summary_func, saver, step_callback):

        def callback_func(**kwargs):
            augmented_parameter_dict = {
                'summary_func': summary_func,
                'saver': saver,
                'session': session
            }

            augmented_parameter_dict.update(kwargs)

            return step_callback(**augmented_parameter_dict)

        return callback_func


class CustomizedOptimizerInterface(ExternalOptimizerInterface):

    def _minimize(self,
                  initial_val,
                  loss_grad_func,
                  single_loss_func,
                  single_grad_func,
                  step_callback,
                  optimizer_kwargs):
        def loss_grad_func_wrapper(x):
            _, loss, grad_left, grad_total, grad_right = loss_grad_func(x)
            return loss, grad_left, grad_total, grad_right

        minimize_args = [loss_grad_func_wrapper,
                         single_loss_func,
                         single_grad_func,
                         initial_val]

        minimize_kwargs = {
            'callback': step_callback
        }
        minimize_kwargs.update(optimizer_kwargs)

        minimize_lbfgs.fmin_l_bfgs(*minimize_args, **minimize_kwargs)


def _accumulate(list_):
    total = 0
    yield total
    for x in list_:
        total += x
        yield total


def _get_shape_tuple(tensor):
    return tuple(dim.value for dim in tensor.get_shape())


def _prod(array):
    # this func serves to calculate the number of elements in a tensor
    # by timing all dimensions together
    prod = 1
    for value in array:
        prod *= value
    return prod


def _compute_gradients(tensor, var_list):
    grads = gradients.gradients(tensor, var_list)
    return [grad if grad is not None else array_ops.zeros_like(var)
            for var, grad in zip(var_list, grads)]
