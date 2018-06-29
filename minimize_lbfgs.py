from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import array, asarray, float64, int32, zeros

import time

import threading
import Queue

# Implement thread class, define cuda Thread for GPU task management.
class cudaThread(threading.Thread):
    def __init__(self, queue, name, m, arr_len):
        threading.Thread.__init__(self)
        self.queue = queue
        self.name = name
        self.thread_stop = False
        self.m = m
        self.arr_len = arr_len
        self.gpu_addr_dict = None
        self.func_dict = None
        self.grid_dim_x_min = None
        self.iter_count = 0

    def run(self):
        import cuda_cal_dk
        while not self.thread_stop:
            try:
                task = self.queue.get(block=True, timeout=8)
            except Queue.Empty:
                self.thread_stop = True
                break
            if self.iter_count == 0:
                a = task[0]
                b = task[1]
                d = task[2]
                cur_iter = task[3]
                self.gpu_addr_dict, self.func_dict, self.grid_dim_x_min = cuda_cal_dk.cal_dk(a, b, d, self.m, self.arr_len, cur_iter)
                self.iter_count += 1
            else:
                d_res = task[0]
                cur_iter = task[1]
                a_add = task[2]
                b_add = task[3]
                cuda_cal_dk.recal_dk(self.gpu_addr_dict, self.func_dict, a_add, b_add, d_res, self.m, self.grid_dim_x_min, cur_iter)
            self.queue.task_done()
            res=self.queue.qsize()
            if res>0:
                print("There are still %d tasks to do" % (res))

    def stop(self):
        self.thread_stop = True

# This function serves to perform gradient calculation and apply them to network. 
# Second order information (Hessian matrix) is incorporated approximated by L-BFGS algorithm.
def fmin_l_bfgs(loss_grad_func,
                single_loss_func,
                single_grad_func,
                x0, m=10, gtol=1e-5, maxiter=100,
                disp=None, callback=None):
    x0 = asarray(x0).ravel()
    n, = x0.shape

    x = array(x0, float64)

    # Define g1 and g2, they take turns to store computation results
    g1 = {'go_before': None,
          'gs': None,
          'go_next': None}

    g2 = {'go_before': None,
          'gs': None,
          'go_next': None}

    # start time counter before the first iter
    start_time = time.time()

    f, g1['go_before'], g1['gs'], g1['go_next'] = loss_grad_func(x)

    assert not np.isnan(f), 'Model diverged with loss = NaN'

    # Define sk and yk for computation results
    sk = []
    yk = []

    n_iterations = 0

    eps_arr = []
    spb_arr = []

    for n_iterations in xrange(maxiter):
        grad_check = 0

        cur_g = g1['gs'] if n_iterations % 2 == 0 else g2['gs']

        dk = np.array(cur_g)
        
        start_time_dk = time.time()

        # Perform computation on GPU to calculate dk
        if n_iterations == 0:
            dk = cur_g * (-1)
        elif n_iterations == 1:

            arr_len = int(cur_g.shape[0])

            # Multithread implementation
            task_q = Queue.Queue(3)
            cudaT1 = cudaThread(task_q, 'thread1', m, arr_len)
            cudaT1.start()
            task_q.put([sk, yk, dk, n_iterations], block=True, timeout=None)
            # Wait for task to complete
            task_q.join()
        else:
            task_q.put([dk, n_iterations, sk[-1], yk[-1]], block=True, timeout=None)
            task_q.join()

        duration_dk = time.time() - start_time_dk

        if n_iterations > 0:
            x = x_next
        step_length = _f_step_length(x, f, cur_g, dk, single_loss_func, single_grad_func)
        # step_length = 0.02
        x_next = x + step_length * dk

        # this is the endpoint of one iter, jot down current timestamp
        duration = time.time() - start_time

        # this part invokes the step_callback function and pass the
        # status information on current step in form of parameters to this func
        if callback is not None:
            callback(step=n_iterations, step_length=step_length, duration=duration_dk, loss_val=f, eps=eps_arr, spb=spb_arr)

        # jot down timestamp at the beginning of one iter
        start_time = time.time()

        if not len(sk) < m:
            del sk[0]
        if not len(yk) < m:
            del yk[0]

        sk.append(x_next - x)

        if n_iterations % 2 == 0:
            f, g2['go_before'], g2['gs'], g2['go_next'] = loss_grad_func(x_next)

            assert not np.isnan(f), 'Model diverged with loss = NaN'

            for grad in g2['gs']:
                if abs(grad) > gtol:
                    break
                else:
                    grad_check += 1
            if grad_check == len(g2['gs']):
                break

            yk.append(g2['go_before'] - g1['go_next'])

        else:
            f, g1['go_before'], g1['gs'], g1['go_next'] = loss_grad_func(x_next)

            assert not np.isnan(f), 'Model diverged with loss = NaN'

            for grad in g1['gs']:
                if abs(grad) > gtol:
                    break
                else:
                    grad_check += 1
            if grad_check == len(g1['gs']):
                break

            yk.append(g1['go_before'] - g2['go_next'])

    return x_next

# Calculate the next descent direction
def _calculate_dk(g, sk, yk, m, cur_iter):
    if cur_iter == 0:
        return g * (-1)

    d = np.array(g)
    k = m if cur_iter > m else cur_iter
    print(d[-20:])
    alpha = []

    for i in xrange(k - 1, -1, -1):
        alpha.append(np.dot(sk[i], d) / np.dot(yk[i], sk[i]))
        d -= alpha[k - 1 - i] * yk[i]

    alpha.reverse()

    for i in xrange(k):
        beta = np.dot(yk[i], d) / np.dot(yk[i], sk[i])
        d += (alpha[i] - beta) * sk[i]

    return d * (-1)

# Linear search to find the optimized step length
def _f_step_length(x_val, f_init, g_init, dk,
                   single_loss_func,
                   single_grad_func,
                   rho=0.1, sigma=0.7, max_step=100):
    alpha_1 = 0
    alpha_2 = max_step
    alpha = 1

    psi_1 = f_init
    psi_1_grad = np.dot(g_init, dk)

    iter_count = 0

    while True:
        if iter_count > 30:
            return alpha

        psi_alpha = single_loss_func(x_val + alpha * dk)

        if psi_alpha - psi_1 > rho * alpha * psi_1_grad:
            alpha_average = alpha_1 + (alpha - alpha_1) / \
                (2 * (1 + (psi_1 - psi_alpha) / ((alpha - alpha_1) * psi_1_grad)))
            
            alpha_2 = alpha
            alpha = alpha_average

            iter_count += 1

            continue

        psi_alpha_grad = np.dot(single_grad_func(x_val + alpha * dk), dk)

        if psi_alpha_grad < sigma * psi_1_grad:
            alpha_average = alpha + (alpha - alpha_1) * psi_alpha_grad / \
                (psi_1_grad - psi_alpha_grad)
            
            alpha_1 = alpha
            alpha = alpha_average
            psi_1 = psi_alpha
            psi_1_grad = psi_alpha_grad

            iter_count += 1

        else:
            return alpha
