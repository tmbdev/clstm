#include "clstm_compute.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

// The NOINLINE attribute is used before all forward_/backward_ steps
// to make execution profiles a little more readable (probably not
// needed).

#ifndef NOINLINE
#define NOINLINE __attribute__((noinline))
#endif

// The host/device directives are only meaningful with CUDACC

#ifdef __CUDACC__
#define ONBOTH __host__ __device__
#define ONDEVICE __device__
#else
#define ONBOTH
#define ONDEVICE
#endif

namespace ocropus {

inline void print2d(TensorRef2 t) {
  for (int i = 0; i < t.dimension(0); i++) {
    for (int j = 0; j < t.dimension(1); j++) {
      std::cerr << std::setw(8) << t(i, j);
    }
    std::cerr << "\n";
  }
}

// We can generate code for different Eigen devices by defining
// the DEVICE macro when compiling this compilation unit.
//
// When no DEVICE is given, we use the Eigen::DefaultDevice
// and default to some of the Eigen::Matrix routines (which
// are faster in some cases).
//
// When a DEVICE is given, we use all Tensor operations.

#ifndef DEVICE
typedef Eigen::DefaultDevice Device;
Eigen::DefaultDevice default_device;
#else
#define CLSTM_ALL_TENSOR
typedef DEVICE Device;
#endif

inline void device_notify(Device *dev, int gpu) {
  static int count = 0;
  if (count > 0) return;
  cerr << "using " << typeid(dev).name() << " gpu: " << gpu << "\n";
  count++;
}

// When compiling with CUDA, we refer to GPUs by integer index outside
// this code. That ensures that none of the rest of CLSTM has to know
// about CUDA or nvcc.

#if defined(CLSTM_CUDA) && defined(__CUDACC__)
#define MAXGPUS 64

using std::unique_ptr;

struct EigenGpu {
  unique_ptr<Eigen::CudaStreamDevice> stream;
  unique_ptr<Eigen::GpuDevice> dev;
};
static EigenGpu devices[MAXGPUS];

Eigen::GpuDevice *gpu_device(int id) {
  using std::cerr;
  using std::endl;
  if (id < 0) return nullptr;
  assert(id < MAXGPUS);
  if (!devices[id].dev) {
    cerr << "initializing GPU " << id << endl;
    assert(id == 0 && "only GPU 0 tested / supported so far");
    auto stream = new Eigen::CudaStreamDevice(/*id*/);
    devices[id].stream.reset(stream);
    devices[id].dev.reset(new Eigen::GpuDevice(stream));
  }
  return devices[id].dev.get();
}
#endif

// Some utility functions for dealing with Eigen indexes and axes.

typedef Eigen::IndexPair<int> IndexPair;
typedef Eigen::array<IndexPair, 1> Axes1;
typedef Eigen::array<ptrdiff_t, 1> Indexes1;
typedef Eigen::array<ptrdiff_t, 2> Indexes2;
typedef Eigen::array<ptrdiff_t, 3> Indexes3;
typedef Eigen::array<ptrdiff_t, 4> Indexes4;

ONBOTH inline Axes1 axispairs(int i, int j) {
  Axes1 result = {IndexPair(i, j)};
  return result;
}

ONBOTH inline Indexes1 indexes(int i) { return Indexes1({i}); }

ONBOTH inline Indexes2 indexes(int i, int j) { return Indexes2({i, j}); }

// Non-linearities. These come in two versions: regular and in-place.
// Note that the regular ones use additive backward-deltas, while the
// in-place ones just modify the deltas in place.

NOINLINE void forward_identity(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v();
}
NOINLINE void forward_sigmoid(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v().sigmoid();
}
NOINLINE void forward_tanh(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v().tanh();
}
NOINLINE void forward_relu(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v().cwiseMax(Float(0));
}
NOINLINE void forward_logmag(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) =
      (x.v().abs() + Float(1)).log() *
      ((x.v() < Float(0)).cast<Float>() * Float(-2) + Float(1));
}
NOINLINE void forward_nonlin(Device *dev, Batch &y, Batch &x, int nl) {
  switch (nl) {
    case LIN:
      forward_identity(dev, y, x);
      break;
    case SIG:
      forward_sigmoid(dev, y, x);
      break;
    case TANH:
      forward_tanh(dev, y, x);
      break;
    case RELU:
      forward_relu(dev, y, x);
      break;
    case LOGMAG:
      forward_logmag(dev, y, x);
      break;
    default:
      abort();
  }
}

NOINLINE void backward_identity(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += y.d();
}
NOINLINE void backward_sigmoid(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += y.v() * (-y.v() + Float(1)) * y.d();
}
NOINLINE void backward_tanh(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += (-y.v() * y.v() + Float(1)) * y.d();
}
NOINLINE void backward_relu(Device *dev, Batch &y, Batch &x) {
  Float zero = 0;
  x.d().device(*dev) += y.d() * (y.v() > zero).cast<Float>();
}
NOINLINE void backward_logmag(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += y.d() * (-y.v().abs()).exp();
}
NOINLINE void backward_nonlin(Device *dev, Batch &y, Batch &x, int nl) {
  switch (nl) {
    case LIN:
      backward_identity(dev, y, x);
      break;
    case SIG:
      backward_sigmoid(dev, y, x);
      break;
    case TANH:
      backward_tanh(dev, y, x);
      break;
    case RELU:
      backward_relu(dev, y, x);
      break;
    case LOGMAG:
      backward_logmag(dev, y, x);
      break;
    default:
      abort();
  }
}

// Forward and backward non-linearities for in-place processing.

NOINLINE void forward_identity0(Device *dev, Batch &y) {
  y.v().device(*dev) = y.v();
}
NOINLINE void forward_sigmoid0(Device *dev, Batch &y) {
  y.v().device(*dev) = y.v().sigmoid();
}
NOINLINE void forward_tanh0(Device *dev, Batch &y) {
  y.v().device(*dev) = y.v().tanh();
}
NOINLINE void forward_relu0(Device *dev, Batch &y) {
  y.v().device(*dev) = y.v().cwiseMax(Float(0));
}
NOINLINE void forward_logmag0(Device *dev, Batch &y) {
  y.v().device(*dev) =
      (y.v().abs() + Float(1)).log() *
      ((y.v() < Float(0)).cast<Float>() * Float(-2) + Float(1));
}
NOINLINE void forward_nonlin0(Device *dev, Batch &y, int nl) {
  switch (nl) {
    case LIN:
      forward_identity0(dev, y);
      break;
    case SIG:
      forward_sigmoid0(dev, y);
      break;
    case TANH:
      forward_tanh0(dev, y);
      break;
    case RELU:
      forward_relu0(dev, y);
      break;
    case LOGMAG:
      forward_logmag0(dev, y);
      break;
    default:
      abort();
  }
}

NOINLINE void backward_identity0(Device *dev, Batch &y) {
  y.d().device(*dev) = y.d();
}
NOINLINE void backward_sigmoid0(Device *dev, Batch &y) {
  y.d().device(*dev) = y.v() * (-y.v() + Float(1)) * y.d();
}
NOINLINE void backward_tanh0(Device *dev, Batch &y) {
  y.d().device(*dev) = (-y.v() * y.v() + Float(1)) * y.d();
}
NOINLINE void backward_relu0(Device *dev, Batch &y) {
  Float zero = 0;
  y.d().device(*dev) = y.d() * (y.v() > zero).cast<Float>();
}
NOINLINE void backward_logmag0(Device *dev, Batch &y) {
  y.d().device(*dev) = y.d() * (-y.v().abs()).exp();
}
NOINLINE void backward_nonlin0(Device *dev, Batch &y, int nl) {
  switch (nl) {
    case LIN:
      backward_identity0(dev, y);
      break;
    case SIG:
      backward_sigmoid0(dev, y);
      break;
    case TANH:
      backward_tanh0(dev, y);
      break;
    case RELU:
      backward_relu0(dev, y);
      break;
    case LOGMAG:
      backward_logmag0(dev, y);
      break;
    default:
      abort();
  }
}
// Full layers with constant offset

#ifndef CLSTM_ALL_TENSOR
#define CBUTFIRST(M) (M).block(0, 1, (M).rows(), (M).cols() - 1)
#define CFIRST(M) (M).col(0)
#endif

NOINLINE void forward_lin1(Device *dev, Batch &y, Params &W1, Batch &x) {
  int n = W1.v.dimension(0);
  int m = W1.v.dimension(1);
  assert(y.rows() == n);
  assert(y.cols() == x.cols());
  assert(x.rows() == m - 1);
#ifdef CLSTM_ALL_TENSOR
  int bs = y.cols();
  Indexes2 offsets{0, 1};
  Indexes2 sizes{n, m - 1};
  Axes1 axes01{IndexPair(1, 0)};
  y.v().device(*dev) = W1.v.map1().contract(x.v(), axes01);
  Indexes2 shape{n, 1};
  Indexes2 bcast{1, bs};
  y.v().device(*dev) += W1.v.off1().reshape(shape).broadcast(bcast);
#else
  y.v.mat() = (W1.v.mat1() * x.v.mat()).colwise() + W1.v.vec1();
#endif
}
NOINLINE void backward_lin1(Device *dev, Batch &y, Params &W1, Batch &x) {
#ifdef CLSTM_ALL_TENSOR
  x.d().device(*dev) += W1.v.map1().contract(y.d(), axispairs(0, 0));
  W1.d.map1().device(*dev) += y.d().contract(x.v(), axispairs(1, 1));
  W1.d.off1().device(*dev) += y.d().sum(indexes(1));
#else
  x.d.mat() += W1.v.mat1().transpose() * y.d.mat();
  W1.d.mat1() += y.d.mat() * x.v.mat().transpose();
  W1.d.vec1() += y.d.mat().rowwise().sum();
#endif
}

// full layers with nonlinearities

NOINLINE void forward_full1(Device *dev, Batch &y, Params &W1, Batch &x,
                            int nl) {
  assert(y.getGpu() < 0 ? typeid(dev) == typeid(&default_device) : true);
  assert(y.getGpu() >= 0 ? typeid(dev) != typeid(&default_device) : true);
  forward_lin1(dev, y, W1, x);
  forward_nonlin0(dev, y, nl);
}

NOINLINE void backward_full1(Device *dev, Batch &y, Params &W1, Batch &x,
                             int nl) {
  backward_nonlin0(dev, y, nl);
  backward_lin1(dev, y, W1, x);
}

// softmax

NOINLINE void forward_softmax(Device *dev, Batch &z, Params &W1, Batch &x) {
  Float (*f)(Float) = limexp;
  int n = W1.v.dimension(0);
  assert(n == z.v.dimension(0));
  assert(n >= 2);
#ifdef CLSTM_ALL_TENSOR
  int bs = x.cols();
  z.v().device(*dev) = W1.v.map1().contract(x.v(), axispairs(1, 0));
  z.v().device(*dev) +=
      W1.v.off1().reshape(indexes(n, 1)).broadcast(indexes(1, bs));
  z.v().device(*dev) = z.v().unaryExpr(f);
  EigenTensor1 sums = z.v().sum(indexes(0));
  z.v().device(*dev) =
      z.v() / sums.reshape(indexes(1, bs)).broadcast(indexes(n, 1));
  ;
#else
  z.v.mat() = (W1.v.mat1() * x.v.mat()).colwise() + W1.v.vec1();
  z.v.mat() = z.v.mat().unaryExpr(f);
  EigenVector sums = z.v.mat().colwise().sum();
  z.v.mat().array().rowwise() /= sums.transpose().array();
#endif
}
NOINLINE void backward_softmax(Device *dev, Batch &z, Params &W1, Batch &x) {
#ifdef CLSTM_ALL_TENSOR
  x.d().device(*dev) = W1.v.map1().contract(z.d(), axispairs(0, 0));
  W1.d.map1().device(*dev) += z.d().contract(x.v(), axispairs(1, 1));
  W1.d.off1().device(*dev) += z.d().sum(indexes(1));
#else
  x.d.mat() = W1.v.mat1().transpose() * z.d.mat();
  W1.d.mat1() += z.d.mat() * x.v.mat().transpose();
  W1.d.vec1() += z.d.mat().rowwise().sum();
#endif
}

// stacking

NOINLINE void forward_stack(Device *dev, Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
  z.v().slice(indexes(0, 0), indexes(nx, bs)).device(*dev) = x.v();
  z.v().slice(indexes(nx, 0), indexes(ny, bs)).device(*dev) = y.v();
}
NOINLINE void backward_stack(Device *dev, Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  x.d().device(*dev) += z.d().slice(indexes(0, 0), indexes(nx, bs));
  y.d().device(*dev) += z.d().slice(indexes(nx, 0), indexes(ny, bs));
}

// stacking with delay

NOINLINE void forward_stack_delay(Device *dev, Batch &z, Batch &x, Sequence &y,
                                  int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
#ifdef CLSTM_ALL_TENSOR
  z.v().slice(indexes(0, 0), indexes(nx, bs)).device(*dev) = x.v();
  if (last >= 0)
    z.v().slice(indexes(nx, 0), indexes(ny, bs)).device(*dev) = y[last].v();
  else
    z.v().slice(indexes(nx, 0), indexes(ny, bs)).device(*dev) =
        y[0].v().constant(0);
#else
  z.v.mat().block(0, 0, nx, bs) = x.v.mat();
  if (last >= 0)
    z.v.mat().block(nx, 0, ny, bs) = y[last].v.mat();
  else
    z.v.mat().block(nx, 0, ny, bs).setZero();
#endif
}
NOINLINE void backward_stack_delay(Device *dev, Batch &z, Batch &x, Sequence &y,
                                   int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
#ifdef CLSTM_ALL_TENSOR
  x.d().device(*dev) += z.d().slice(indexes(0, 0), indexes(nx, bs));
  if (last >= 0)
    y[last].d().device(*dev) += z.d().slice(indexes(nx, 0), indexes(ny, bs));
#else
  x.d.mat() += z.d.mat().block(0, 0, nx, bs);
  if (last >= 0) y[last].d.mat() += z.d.mat().block(nx, 0, ny, bs);
#endif
}

// reverse sequences

NOINLINE void forward_reverse(Device *dev, Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) y[N - i - 1] = x[i];
}
NOINLINE void backward_reverse(Device *dev, Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) x[N - i - 1].d().device(*dev) += y[i].d();
}

// switch time and batch

NOINLINE void forward_btswitch(Device *dev, Sequence &y, Sequence &x) {
  TensorMap4 y4 = y.map4();
  TensorMap4 x4 = x.map4();
  // dimensions are: (feature, batch, 2, time)
  assert(y4.dimension(0) == x4.dimension(0));
  assert(y4.dimension(1) == x4.dimension(3));
  assert(y4.dimension(2) == 2);
  assert(y4.dimension(3) == x4.dimension(1));

  Indexes3 axes{0, 2, 1};
  y4.chip(0, 2).device(*dev) = x4.chip(0, 2).shuffle(axes);
}
NOINLINE void backward_btswitch(Device *dev, Sequence &y, Sequence &x) {
  TensorMap4 y4 = y.map4();
  TensorMap4 x4 = x.map4();
  assert(y4.dimension(0) == x4.dimension(0));
  assert(y4.dimension(1) == x4.dimension(3));
  assert(y4.dimension(2) == 2);
  assert(y4.dimension(3) == x4.dimension(1));

  Indexes3 axes{0, 2, 1};
  x4.chip(1, 2).device(*dev) += y4.chip(1, 2).shuffle(axes);
}

// stacking neighboring batches

NOINLINE void forward_batchstack(Device *dev, Sequence &y, Sequence &x, int pre,
                                 int post) {
  TensorMap4 y4 = y.map4();
  TensorMap4 x4 = x.map4();
  // dimensions are: (feature, batch, 2, time)
  int d = x4.dimension(0);
  int bs = x4.dimension(1);
  int size = x4.dimension(3);
  int copies = pre + post + 1;
  assert(y4.dimension(0) == copies * d);
  assert(y4.dimension(1) == bs);
  assert(y4.dimension(2) == 2);
  assert(y4.dimension(3) == x4.dimension(3));
  y4.device(*dev) = y4.constant(Float(0));
  for (int k = -pre; k <= post; k++) {
    int source = max(k, 0);
    int dest = max(-k, 0);
    int crimp = abs(k);
    Indexes4 source_offsets{0, source, 0, 0};
    Indexes4 dest_offsets{d * (pre + k), dest, 0, 0};
    Indexes4 sizes{d, bs - crimp, 1, size};
    y4.slice(dest_offsets, sizes).device(*dev) =
        x4.slice(source_offsets, sizes);
  }
}
NOINLINE void backward_batchstack(Device *dev, Sequence &y, Sequence &x,
                                  int pre, int post) {
  TensorMap4 y4 = y.map4();
  TensorMap4 x4 = x.map4();
  // dimensions are: (feature, batch, 2, time)
  int d = x4.dimension(0);
  int bs = x4.dimension(1);
  int size = x4.dimension(3);
  int copies = pre + post + 1;
  assert(y4.dimension(0) == copies * d);
  assert(y4.dimension(1) == bs);
  assert(y4.dimension(2) == 2);
  assert(y4.dimension(3) == x4.dimension(3));
  // x4.chip(1,2).device(*dev) = x4.chip(1,2).constant(Float(0));
  for (int k = -pre; k <= post; k++) {
    int source = max(k, 0);
    int dest = max(-k, 0);
    int crimp = abs(k);
    Indexes4 source_offsets{0, source, 1, 0};
    Indexes4 dest_offsets{d * (pre + k), dest, 1, 0};
    Indexes4 sizes{d, bs - crimp, 1, size};
    x4.slice(source_offsets, sizes).device(*dev) +=
        y4.slice(dest_offsets, sizes);
  }
}

// combine the delayed gated state with the gated input

NOINLINE void forward_statemem(Device *dev, Batch &state, Batch &ci, Batch &gi,
                               Sequence &states, int last, Batch &gf) {
  state.v().device(*dev) = ci.v() * gi.v();
  if (last >= 0) state.v().device(*dev) += gf.v() * states[last].v();
}
NOINLINE void backward_statemem(Device *dev, Batch &state, Batch &ci, Batch &gi,
                                Sequence &states, int last, Batch &gf) {
  if (last >= 0) states[last].d().device(*dev) += state.d() * gf.v();
  if (last >= 0) gf.d().device(*dev) += state.d() * states[last].v();
  gi.d().device(*dev) += state.d() * ci.v();
  ci.d().device(*dev) += state.d() * gi.v();
}

// linear gated output

NOINLINE void forward_gate(Device *dev, Batch &out, Batch &nlstate, Batch &go) {
  out.v().device(*dev) = nlstate.v() * go.v();
}
NOINLINE void backward_gate(Device *dev, Batch &out, Batch &nlstate,
                            Batch &go) {
  go.d().device(*dev) += nlstate.v() * out.d();
  nlstate.d().device(*dev) += go.v() * out.d();
}

// nonlinear gated output

NOINLINE void forward_nonlingate(Device *dev, Batch &out, Batch &state,
                                 Batch &go, int nl) {
  BatchStorage temp;
  temp.setGpu(out.getGpu());
  temp.resize(out.rows(), out.cols());
  forward_nonlin(dev, (Batch &)temp, state, nl);
  forward_gate(dev, out, (Batch &)temp, go);
}

NOINLINE void backward_nonlingate(Device *dev, Batch &out, Batch &state,
                                  Batch &go, int nl) {
  BatchStorage temp;
  temp.setGpu(out.getGpu());
  temp.resize(out.rows(), out.cols());
  forward_nonlin(dev, (Batch &)temp, state, nl);
  backward_gate(dev, out, (Batch &)temp, go);
  backward_nonlin(dev, (Batch &)temp, state, nl);
}

NOINLINE void fill(Device *dev, TensorMap2 &a, Float value) {
  a.device(*dev) = a.constant(value);
}

NOINLINE void clip_gradient(Device *dev, Batch &x, Float clip) {
  if (clip >= 1e6) return;
  assert(clip > 0);
  x.d().device(*dev) = x.d().cwiseMin(clip);
  x.d().device(*dev) = x.d().cwiseMax(-clip);
}

NOINLINE void sgd_update(Device *dev, Params &params, Float lr, Float mom) {
  params.v().device(*dev) += params.d() * lr;
  params.d().device(*dev) = params.d() * mom;
}
}
