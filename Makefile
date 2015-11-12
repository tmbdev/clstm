clstm_compute_cuda.o: clstm_compute_cuda.cc
	scons clstm_compute_cuda.o
test.o: test.cu
	/usr/local/cuda/bin/nvcc --std=c++11 -x cu -DEIGEN_USE_GPU --expt-relaxed-constexpr -I/usr/local/include/eigen3 -c test.cu
DONE: Dockerfile
	docker build -t tmbdev/ubuntu-cuda .
	touch DONE
