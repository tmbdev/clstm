#define DEVICE Eigen::GpuDevice
#ifndef EIGEN_USE_GPU
#error "EIGEN_USE_GPU not defined"
#endif
#ifndef __CUDACC__
#error "not compiling in CUDA mode"
#endif

#include "clstm_compute.cc"
