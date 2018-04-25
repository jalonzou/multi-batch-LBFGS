from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import array, asarray, float64, int32, zeros

import time


def fmin_l_bfgs(loss_grad_func,
                single_loss_func,
                single_grad_func,
                x0, m=10, gtol=1e-5, maxiter=100,
                disp=None, callback=None):
    x0 = asarray(x0).ravel()
    n, = x0.shape

    x = array(x0, float64)

    #x_next = zeros((n,), float64)
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

    sk = []
    yk = []

    n_iterations = 0

    for n_iterations in xrange(maxiter):
        grad_check = 0

        cur_g = g1['gs'] if n_iterations % 2 == 0 else g2['gs']

        start_time_dk = time.time()
        dk = _calculate_dk(cur_g, sk, yk, m, n_iterations)
        duration_dk = time.time() - start_time_dk

        if n_iterations > 0:
            x = x_next
        step_length = _f_step_length(x, f, cur_g, dk, single_loss_func, single_grad_func)
        # step_length = 1.0
        x_next = x + step_length * dk
        # x_next = x + dk

        # this is the endpoint of one iter, jot down current timestamp
        duration = time.time() - start_time

        # this part invokes the step_callback function and pass the
        # status information on current step in form of parameters to this func
        if callback is not None:
            callback(step=n_iterations, step_length=step_length, duration=duration, loss_val=f)

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


def _calculate_dk(g, sk, yk, m, cur_iter):
    if cur_iter == 0:
        return g * (-1)

    d = np.array(g)
    k = m if cur_iter > m else cur_iter

    alpha = []

    for i in xrange(k - 1, -1, -1):
        alpha.append(np.dot(sk[i], d) / np.dot(yk[i], sk[i]))
        # print(sk[i][:10], yk[i][:10])
        d -= alpha[k - 1 - i] * yk[i]

    alpha.reverse()

    # d *= np.dot(sk[k-1], yk[k-1]) / np.dot(yk[k-1], yk[k-1])

    for i in xrange(k):
        beta = np.dot(yk[i], d) / np.dot(yk[i], sk[i])
        d += (alpha[i] - beta) * sk[i]

    # print(d[:10])

    return d * (-1)


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
            # print('con1')
            # alpha_2 = alpha
            # alpha = (alpha_1 + alpha) / 2

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
            # alpha_1 = alpha
            # alpha = 2*alpha if 2*alpha <= (alpha_1+alpha_2)/2 else (alpha_1+alpha_2)/2
            # print('con2')

        else:
            return alpha
