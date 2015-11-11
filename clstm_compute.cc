#include <memory>
#include "clstm_compute.h"
#include <unsupported/Eigen/CXX11/Tensor>

// FIXME: factor out nonlinearities

namespace ocropus {

#ifndef DEVICE
typedef Eigen::DefaultDevice Device;
Eigen::DefaultDevice default_device;

#else
typedef DEVICE Device;
#endif

#ifdef __CUDACC__
#define ONBOTH __host__ __device__
#define ONDEVICE __device__
#define MAXGPUS 16

using std::unique_ptr;

struct EigenGpu {
  unique_ptr<Eigen::CudaStreamDevice> stream;
  unique_ptr<Eigen::GpuDevice> dev;
};
static EigenGpu devices[MAXGPUS];

Eigen::GpuDevice *gpu_device(int id) {
  if (id<0) return nullptr;
  assert(id<MAXGPUS);
  if (!devices[id].dev) {
    auto stream = new Eigen::CudaStreamDevice(/*id*/);
    devices[id].stream.reset(stream);
    devices[id].dev.reset(new Eigen::GpuDevice(stream));
  }
  return devices[id].dev.get();
}

#else
#define ONBOTH
#define ONDEVICE
#endif

typedef Eigen::IndexPair<int> IndexPair;
typedef Eigen::array<IndexPair, 1> Axes1;
typedef Eigen::array<ptrdiff_t, 1> Indexes1;
typedef Eigen::array<ptrdiff_t, 2> Indexes2;

ONBOTH inline Axes1 axispairs(int i, int j) {
  Axes1 result = {IndexPair(i, j)};
  return result;
}

ONBOTH inline Indexes1 indexes(int i) {
  return Indexes1({i});
}

ONBOTH inline Indexes2 indexes(int i, int j) {
  return Indexes2({i, j});
}

// Non-linearities. These can either be run "in place"
// on the output of a linear layer, or as a regular
// step. When run "in place", there is a separate backwards
// step that, unlike regular backwards steps, doesn't add
// to the delta on the input but just sets it.

void forward_identity(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v();
}
void forward_sigmoid(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v().sigmoid();
}
void forward_tanh(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v().tanh();
}
void forward_relu(Device *dev, Batch &y, Batch &x) {
  y.v().device(*dev) = x.v().cwiseMax(Float(0));
}
void forward_nonlin(Device *dev, Batch &y, Batch &x, int nl) {
  switch(nl) {
  case LIN: forward_identity(dev, y, x); break;
  case SIG: forward_sigmoid(dev, y, x); break;
  case TANH: forward_tanh(dev, y, x); break;
  case RELU: forward_relu(dev, y, x); break;
  default: abort();
  }
}

void backward_identity(Device *dev, Batch &y) {
  y.d().device(*dev) = y.d();
}
void backward_sigmoid(Device *dev, Batch &y) {
  y.d().device(*dev) = y.v() * (-y.v()+Float(1)) * y.d();
}
void backward_tanh(Device *dev, Batch &y) {
  y.d().device(*dev) = (-y.v()*y.v() + Float(1)) * y.d();
}
void backward_relu(Device *dev, Batch &y) {
  Float zero = 0;
  y.d().device(*dev) = y.d() * (y.v()>zero).cast<Float>();
}
void backward_nonlin(Device *dev, Batch &y, int nl) {
  switch(nl) {
  case LIN: backward_identity(dev, y); break;
  case SIG: backward_sigmoid(dev, y); break;
  case TANH: backward_tanh(dev, y); break;
  case RELU: backward_relu(dev, y); break;
  default: abort();
  }
}

void backward_identity(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += y.d();
}
void backward_sigmoid(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += y.v() * (-y.v()+Float(1)) * y.d();
}
void backward_tanh(Device *dev, Batch &y, Batch &x) {
  x.d().device(*dev) += (-y.v()*y.v() + Float(1)) * y.d();
}
void backward_relu(Device *dev, Batch &y, Batch &x) {
  Float zero = 0;
  x.d().device(*dev) += y.d() * (y.v()>zero).cast<Float>();
}
void backward_nonlin(Device *dev, Batch &y, Batch &x, int nl) {
  switch(nl) {
  case LIN: backward_identity(dev, y, x); break;
  case SIG: backward_sigmoid(dev, y, x); break;
  case TANH: backward_tanh(dev, y, x); break;
  case RELU: backward_relu(dev, y, x); break;
  default: abort();
  }
}

// full layers with constant offset

void forward_lin1(Device *dev, Batch &y, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = x.v.dimension(1);
  assert(y.rows() == n);
  assert(y.cols() == x.cols());
  assert(x.rows() == m-1);
  Indexes2 offsets{0, 1};
  Indexes2 sizes{n, m-1};
  Axes1 axes01{IndexPair(1,0)};
  y.v().device(*dev) = W1.v().slice(offsets, sizes).contract(x.v(), axes01);
  Indexes2 shape{n, 1};
  Indexes2 bcast{1, bs};
  y.v().device(*dev) += W1.v().chip(0, 1).reshape(shape).broadcast(bcast);
}
void backward_lin1(Device *dev, Batch &y, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  x.d().device(*dev) += W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(y.d(), axispairs(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)).device(*dev) += y.d().contract(x.v(), axispairs(1, 1));
  W1.d().chip(0, 1).device(*dev) += y.d().sum(indexes(1));
}

// full layers with nonlinearities

void forward_full1(Device *dev, Batch &y, Params &W1, Batch &x, Nonlin nl) {
  forward_lin1(dev, y, W1, x);
  forward_nonlin(dev, y, y, nl);
}


void backward_full1(Device *dev, Batch &y, Params &W1, Batch &x, Nonlin nl) {
  backward_nonlin(dev, y, nl);
  backward_lin1(dev, y, W1, x);
}

// softmax

void forward_softmax(Device *dev, Batch &z, Params &W1, Batch &x) {
  Float (*f)(Float) = limexp;
  int n = W1.v.dimension(0);
  int m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  assert(n == z.v.dimension(0));
  assert(n >= 2);
  z.v().device(*dev) = (W1.v()
             .slice(indexes(0, 1), indexes(n, m - 1))
             .contract(x.v(), axispairs(1, 0)) +
         W1.v().chip(0, 1).reshape(indexes(n, 1)).broadcast(indexes(1, bs)))
            .unaryExpr(f);
  EigenTensor1 sums = z.v().sum(indexes(0));
  assert(sums.dimension(0)==bs);
  z.v().device(*dev) = z.v() / sums.reshape(indexes(1,bs)).broadcast(indexes(n,1));;
}
void backward_softmax(Device *dev, Batch &z, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  x.d().device(*dev) = W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(z.d(), axispairs(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)).device(*dev) += z.d().contract(x.v(), axispairs(1, 1));
  W1.d().chip(0, 1).device(*dev) += z.d().sum(indexes(1));
}

// stacking

void forward_stack(Device *dev, Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
  z.v().slice(indexes(0, 0), indexes(nx, bs)).device(*dev) = x.v();
  z.v().slice(indexes(nx, 0), indexes(ny, bs)).device(*dev) = y.v();
}
void backward_stack(Device *dev, Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  x.d().device(*dev) += z.d().slice(indexes(0, 0), indexes(nx, bs));
  y.d().device(*dev) += z.d().slice(indexes(nx, 0), indexes(ny, bs));
}

// stacking with delay

void forward_stack_delay(Device *dev, Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
  z.v().slice(indexes(0, 0), indexes(nx, bs)).device(*dev) = x.v();
  if (last >= 0)
    z.v().slice(indexes(nx, 0), indexes(ny, bs)).device(*dev) = y[last].v();
  else
    z.v().slice(indexes(nx, 0), indexes(ny, bs)).setZero();
}
void backward_stack_delay(Device *dev, Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  x.d().device(*dev) += z.d().slice(indexes(0, 0), indexes(nx, bs));
  if (last >= 0) y[last].d().device(*dev) += z.d().slice(indexes(nx, 0), indexes(ny, bs));
}

// reverse sequences

void forward_reverse(Device *dev, Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) y[N - i - 1] = x[i];
}
void backward_reverse(Device *dev, Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) x[N - i - 1].d().device(*dev) += y[i].d();
}

// combine the delayed gated state with the gated input

void forward_statemem(Device *dev, Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf) {
  state.v().device(*dev) = ci.v() * gi.v();
  if (last >= 0) state.v().device(*dev) += gf.v() * states[last].v();
}
void backward_statemem(Device *dev, Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf) {
  if (last >= 0) states[last].d().device(*dev) += state.d() * gf.v();
  if (last >= 0) gf.d().device(*dev) += state.d() * states[last].v();
  gi.d().device(*dev) += state.d() * ci.v();
  ci.d().device(*dev) += state.d() * gi.v();
}

// linear gated output

void forward_gate(Device *dev, Batch &out, Batch &nlstate, Batch &go) {
  out.v().device(*dev) = nlstate.v() * go.v();
}
void backward_gate(Device *dev, Batch &out, Batch &nlstate, Batch &go) {
  go.d().device(*dev) += nlstate.v() * out.d();
  nlstate.d().device(*dev) += go.v() * out.d();
}

// nonlinear gated output

void forward_nonlingate(Device *dev, Batch &out, Batch &state, Batch &go, int nl) {
  Batch temp;
  temp.resize(out.rows(), out.cols());
  forward_nonlin(dev, temp, state, nl);
  forward_gate(dev, out, temp, go);
}

void backward_nonlingate(Device *dev, Batch &out, Batch &state, Batch &go, int nl) {
  Batch temp;
  temp.resize(out.rows(), out.cols());
  forward_nonlin(dev, temp, state, nl);
  backward_gate(dev, out, temp, go);
  backward_nonlin(dev, temp, state, nl);
}

void fill(Device *dev, TensorMap2 &a, Float value) {
  a.device(*dev) = a.constant(value);
}

void sgd_update(Device *dev, Params &params, Float lr, Float mom) {
  params.v().device(*dev) += params.d() * lr;
  params.d().device(*dev) = params.d() * mom;
}

}

