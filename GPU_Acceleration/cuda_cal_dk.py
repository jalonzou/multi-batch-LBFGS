import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit
# from pycuda.autoinit import context
from pycuda.compiler import SourceModule

import numpy as np
import copy
import time

# CUDA implementation of dk calculation
def cal_dk(a, b, d, m, arr_len, cur_iter):
	m = m if cur_iter > m else cur_iter

	reduce_1_scale = 1000
	reduce_1_iter = int(arr_len / reduce_1_scale)
	extra_ker_id = arr_len % reduce_1_scale

	grid_dim_x_min = int(arr_len / 1024)
	if grid_dim_x_min * 1024 < arr_len:
	  grid_dim_x_min += 1


	# Declare structure to store computation results from GPU                                                                                    
	t = np.zeros((arr_len), dtype=np.float32)
	a_t = np.zeros((arr_len), dtype=np.float32)

	sum_a = np.zeros((reduce_1_scale), dtype=np.float32)
	sum_b = np.zeros((reduce_1_scale), dtype=np.float32)
	sum_t = np.zeros((reduce_1_scale), dtype=np.float32)

	sum_alpha = np.zeros((1), dtype=np.float32)
	sum_t_res = np.zeros((1), dtype=np.float32)

	run_para = np.array([arr_len, reduce_1_iter, extra_ker_id, reduce_1_scale], dtype=np.int32)

	# GPU resource allocation
	a_group = []
	b_group = []
	c_group = []
	t_res_group = []
	alpha_group = []

	for i in xrange(m):
	  a_group.append(cuda.mem_alloc(a[i].nbytes))
	  b_group.append(cuda.mem_alloc(b[i].nbytes))
	  t_res_group.append(cuda.mem_alloc(sum_t_res.nbytes))
	  alpha_group.append(cuda.mem_alloc(sum_alpha.nbytes))

	t_gpu = cuda.mem_alloc(t.nbytes)
	d_gpu = cuda.mem_alloc(d.nbytes)
	a_t_gpu = cuda.mem_alloc(a_t.nbytes)

	sum_a_gpu = cuda.mem_alloc(sum_a.nbytes)
	sum_b_gpu = cuda.mem_alloc(sum_b.nbytes)
	sum_t_gpu = cuda.mem_alloc(sum_t.nbytes)

	run_para_gpu = cuda.mem_alloc(run_para.nbytes)

	# print t[-10:]

	# Kernel implementaion
	mod = SourceModule("""
	  __global__ void stepone(float *a, float *a_t, float *b, float *t, float *d, int *run_para)
	  {
	    int idx = (blockIdx.y*gridDim.x+blockIdx.x) * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	    if(idx < run_para[0]){
	      t[idx] = a[idx]*b[idx];
	      a_t[idx] = a[idx]*d[idx];
	    }
	  }

	  __global__ void stepone2(float *b, float *t, float *d, int *run_para)
	  {
	    int idx = (blockIdx.y*gridDim.x+blockIdx.x) * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	    if(idx < run_para[0]){
	      t[idx] = b[idx]*d[idx];
	    }
	  }

	  __global__ void steptwo(float *a_t, float *t, float *sum_a, float *sum_t, int *run_para)
	  {
	    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	    sum_a[idx] = 0;
	    sum_t[idx] = 0;
	    int num_iter = run_para[1];
	    for(int i=idx*num_iter; i<(idx+1)*num_iter; i++){
	      sum_a[idx] += a_t[i]; 
	      sum_t[idx] += t[i];
	    }
	    if(idx < run_para[2]){
	      sum_a[idx] += a_t[run_para[0]-1-idx];
	      sum_t[idx] += t[run_para[0]-1-idx];
	    }
	  }

	  __global__ void steptwo2(float *t, float *sum_b, int *run_para)
	  {
	    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	    sum_b[idx] = 0;
	    int num_iter = run_para[1];
	    for(int i=idx*num_iter; i<(idx+1)*num_iter; i++){
	      sum_b[idx] += t[i];
	    }
	    if(idx < run_para[2]){
	      sum_b[idx] += t[run_para[0]-1-idx];
	    }
	  }

	  __global__ void stepthree(float *sum_a, float *sum_t, float *sum_t_res, float *sum_alpha, int *run_para)
	  {
	    int idx = threadIdx.x;
	    int reduce_scale = run_para[3];
	    sum_t_res[0] = sum_t[idx*reduce_scale];
	    for(int i=idx*reduce_scale+1; i<(idx+1)*reduce_scale; i++){
	      sum_a[0] += sum_a[i];
	      sum_t_res[0] += sum_t[i];
	    }
	    sum_alpha[0] = sum_a[0]/sum_t_res[0];
	  }

	  __global__ void stepthree2(float *sum_alpha, float *sum_b, float *sum_t_res, int *run_para)
	  {
	    int idx = threadIdx.x;
	    int reduce_scale = run_para[3];
	    for(int i=idx*reduce_scale+1; i<(idx+1)*reduce_scale; i++){
	      sum_b[0] += sum_b[i];
	    }
	    sum_b[1] = sum_b[0]/sum_t_res[0];
	    sum_t_res[0] = sum_alpha[0]-sum_b[1];
	  }

	  __global__ void stepfour(float *sum_alpha, float *b, float *d, int *run_para)
	  {
	    int idx = (blockIdx.y*gridDim.x+blockIdx.x) * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	    if(idx < run_para[0]){
	      d[idx] -= b[idx] * sum_alpha[0];
	    }
	  }

	  __global__ void stepfour2(float *sum_t_res, float *a, float *d, int *run_para)
	  {
	    int idx = (blockIdx.y*gridDim.x+blockIdx.x) * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	    if(idx < run_para[0]){
	      d[idx] += a[idx] * sum_t_res[0];
	    }
	  }

	  __global__ void stepfour3(float *sum_t_res, float *a, float *d, int *run_para)
	  {
	    int idx = (blockIdx.y*gridDim.x+blockIdx.x) * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	    if(idx < run_para[0]){
	      d[idx] += a[idx] * sum_t_res[0];
	      d[idx] *= (-1);
	    }
	  }
	  """)

	# Obtain kernel function
	func1 = mod.get_function("stepone")
	func2 = mod.get_function("steptwo")
	func3 = mod.get_function("stepthree")
	func4 = mod.get_function("stepfour")

	func12 = mod.get_function("stepone2")
	func22 = mod.get_function("steptwo2")
	func32 = mod.get_function("stepthree2")
	func42 = mod.get_function("stepfour2")

	func43 = mod.get_function("stepfour3")

	# Send computation data to GPU
	for i in xrange(m):
		cuda.memcpy_htod(a_group[i], a[i].astype(np.float32))
		cuda.memcpy_htod(b_group[i], b[i])
		cuda.memcpy_htod(alpha_group[i], sum_alpha)
		cuda.memcpy_htod(t_res_group[i], sum_t_res)

	cuda.memcpy_htod(d_gpu, d)

	cuda.memcpy_htod(run_para_gpu, run_para)

	# Perform computation, execute kernel on GPU
	for i in xrange(m-1, -1, -1):
	  func1(a_group[i], a_t_gpu, b_group[i], t_gpu, d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))
	  func2(a_t_gpu, t_gpu, sum_a_gpu, sum_t_gpu, run_para_gpu, block=(100,1,1), grid=(10,1))
	  func3(sum_a_gpu, sum_t_gpu, t_res_group[i], alpha_group[i], run_para_gpu, block=(1,1,1))
	  func4(alpha_group[i], b_group[i], d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))

	for i in xrange(m):
	  func12(b_group[i], t_gpu, d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))
	  func22(t_gpu, sum_b_gpu, run_para_gpu, block=(100,1,1), grid=(10,1))
	  func32(alpha_group[i], sum_b_gpu, t_res_group[i], run_para_gpu, block=(1,1,1))
	  if i == m-1:
	    func43(t_res_group[i], a_group[i], d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))
	    break
	  func42(t_res_group[i], a_group[i], d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))

	# Collect results from GPU
	cuda.memcpy_dtoh(d, d_gpu)

	# print d[:20]
	gpu_addr_dict = {
	'a_group': a_group,
	'b_group': b_group,
	'd_gpu': d_gpu,
	't_gpu': t_gpu,
	'a_t_gpu': a_t_gpu,
	't_res_group': t_res_group,
	'alpha_group': alpha_group,
	'sum_a_gpu': sum_a_gpu,
	'sum_b_gpu': sum_b_gpu,
	'sum_t_gpu': sum_t_gpu,
	'run_para_gpu': run_para_gpu
	}

	func_dict = {
	'func1': func1,
	'func2': func2,
	'func3': func3,
	'func4': func4,
	'func12': func12,
	'func22': func22,
	'func32': func32,
	'func42': func42,
	'func43': func43
	}
	return gpu_addr_dict, func_dict ,grid_dim_x_min

def recal_dk(gpu_addr_dict, func_dict, a_add, b_add, d_res, m, grid_dim_x_min, cur_iter):
	m = m if cur_iter > m else cur_iter

	# get the computation data
	a_group = gpu_addr_dict['a_group']
	b_group = gpu_addr_dict['b_group']
	d_gpu = gpu_addr_dict['d_gpu']
	t_gpu = gpu_addr_dict['t_gpu']
	a_t_gpu = gpu_addr_dict['a_t_gpu']
	t_res_group = gpu_addr_dict['t_res_group']
	alpha_group = gpu_addr_dict['alpha_group']
	sum_a_gpu = gpu_addr_dict['sum_a_gpu']
	sum_b_gpu = gpu_addr_dict['sum_b_gpu']
	sum_t_gpu = gpu_addr_dict['sum_t_gpu']
	run_para_gpu = gpu_addr_dict['run_para_gpu']

	init_at = np.zeros((1), dtype=np.float32)

	if cur_iter > 10:
		del a_group[0]
		del b_group[0]
	else:
		alpha_group.append(cuda.mem_alloc(init_at.nbytes))
		t_res_group.append(cuda.mem_alloc(init_at.nbytes))

	# Send computation data to GPU
	a_add = a_add.astype(np.float32)
	a_group.append(cuda.mem_alloc(a_add.nbytes))
	b_group.append(cuda.mem_alloc(b_add.nbytes))
	cuda.memcpy_htod(a_group[-1], a_add)
	cuda.memcpy_htod(b_group[-1], b_add)

	cuda.memcpy_htod(d_gpu, d_res)

	for i in xrange(m):
		cuda.memcpy_htod(alpha_group[i], init_at)
		cuda.memcpy_htod(t_res_group[i], init_at)

	# Perform computation on GPU
	for i in xrange(m-1, -1, -1):
		func_dict['func1'](a_group[i], a_t_gpu, b_group[i], t_gpu, d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))
		func_dict['func2'](a_t_gpu, t_gpu, sum_a_gpu, sum_t_gpu, run_para_gpu, block=(100,1,1), grid=(10,1))
		func_dict['func3'](sum_a_gpu, sum_t_gpu, t_res_group[i], alpha_group[i], run_para_gpu, block=(1,1,1))
		func_dict['func4'](alpha_group[i], b_group[i], d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))

	# Perform computation on GPU
	for i in xrange(m):
	  func_dict['func12'](b_group[i], t_gpu, d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))
	  func_dict['func22'](t_gpu, sum_b_gpu, run_para_gpu, block=(100,1,1), grid=(10,1))
	  func_dict['func32'](alpha_group[i], sum_b_gpu, t_res_group[i], run_para_gpu, block=(1,1,1))
	  if i == m-1:
	    func_dict['func43'](t_res_group[i], a_group[i], d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))
	    break
	  func_dict['func42'](t_res_group[i], a_group[i], d_gpu, run_para_gpu, block=(32,32,1), grid=(grid_dim_x_min,1))

	# Collect results from GPU
	cuda.memcpy_dtoh(d_res, d_gpu)