#ifndef EIGEN_USE_GPU
#error no EIGEN_USE_GPU
#endif
#ifndef __CUDACC__
#error no CUDA
#endif

#define HOSTDEV __host__ __device__
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::TensorMap;
using Eigen::Tensor;
typedef TensorMap<Tensor<float,2>> TensorMap2;
typedef Eigen::IndexPair<int> IndexPair;
typedef Eigen::array<IndexPair, 1> Axes1;
typedef Eigen::array<ptrdiff_t, 1> Indexes1;
typedef Eigen::array<ptrdiff_t, 2> Indexes2;

Eigen::GpuDevice *dev;

void spread_add(TensorMap2 y, TensorMap2 x) {
  Indexes2 vshape{1,1};
  Indexes2 bcast{1,1};
  //y.device(*dev) += x.reshape(bcast);
  y.device(*dev) += x.broadcast(bcast);
  //y.device(*dev) += x.reshape(vshape).broadcast(bcast);
  //y.device(*dev) += x.chip(0, 1).reshape(vshape).broadcast(bcast);
}
