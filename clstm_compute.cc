#include "clstm_compute.h"

// FIXME: factor out nonlinearities

namespace ocropus {

inline Eigen::array<Eigen::IndexPair<int>, 1> axispairs(int i, int j) {
  Eigen::array<Eigen::IndexPair<int>, 1> result = {Eigen::IndexPair<int>(i, j)};
  return result;
}

inline Eigen::array<ptrdiff_t, 1> indexes(int i) { 
  return Eigen::array<ptrdiff_t, 1>({i}); 
}

inline Eigen::array<ptrdiff_t, 2> indexes(int i, int j) {
  return Eigen::array<ptrdiff_t, 2>({i, j});
}

typedef Float (*FloatFun)(Float);

struct Nonlinearity {
  FloatFun nonlin;
  FloatFun yderiv;
  FloatFun xderiv;
};

Nonlinearity nonlinearities[] = {
  {
    [](Float x) { return x; },
    [](Float y) { return Float(1); },
    [](Float x) { return Float(1); },
  },
  {
    [](Float x) { return sigmoid(x); },
    [](Float y) { return y * (1-y); },
    [](Float x) { Float y = sigmoid(x); return y * (1-y); }
  },
  {
    [](Float x) { return tanh(x); },
    [](Float y) { return 1 - y*y; },
    [](Float x) { Float y = tanh(x); return 1 - y*y; }
  },
  {
    [](Float x) { return x<0?0:x; },
    [](Float y) { return Float(y<=0?0:1); },
    [](Float x) { return Float(x<=0?0:1); }
  }
};

// full layers with constant offset

void forward_full1(Device *dev, Batch &y, Params &W1, Batch &x, Nonlin nl) {
  Float (*f)(Float) = nonlinearities[nl].nonlin;
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = x.v.dimension(1);
  assert(y.rows() == n);
  assert(y.cols() == x.cols());
  assert(x.rows() == m-1);
  y.v =
      (W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(x.v(), axispairs(1, 0)) +
       W1.v().chip(0, 1).reshape(indexes(n, 1)).broadcast(indexes(1, bs))).unaryExpr(f);
}
void backward_full1(Device *dev, Batch &y, Params &W1, Batch &x, Nonlin nl) {
  Float (*g)(Float) = nonlinearities[nl].yderiv;
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  EigenTensor2 temp = y.v().unaryExpr(g) * y.d();
  x.d += W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(temp, axispairs(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)) += temp.contract(x.v(), axispairs(1, 1));
  W1.d().chip(0, 1) += temp.sum(indexes(1));
}

// softmax

void forward_softmax(Device *dev, Batch &z, Params &W1, Batch &x) {
  Float (*f)(Float) = limexp;
  int n = W1.v.dimension(0);
  int m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  assert(n == z.v.dimension(0));
  assert(n >= 2);
  z.v = (W1.v()
             .slice(indexes(0, 1), indexes(n, m - 1))
             .contract(x.v(), axispairs(1, 0)) +
         W1.v().chip(0, 1).reshape(indexes(n, 1)).broadcast(indexes(1, bs)))
            .unaryExpr(f);
  EigenTensor1 sums = z.v().sum(indexes(0));
  assert(sums.dimension(0)==bs);
  z.v = z.v() / sums.reshape(indexes(1,bs)).broadcast(indexes(n,1));;
}
void backward_softmax(Device *dev, Batch &z, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  x.d = W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(z.d(), axispairs(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)) += z.d().contract(x.v(), axispairs(1, 1));
  W1.d().chip(0, 1) += z.d().sum(indexes(1));
}

// stacking

void forward_stack(Device *dev, Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
  z.v().slice(indexes(0, 0), indexes(nx, bs)) = x.v();
  z.v().slice(indexes(nx, 0), indexes(ny, bs)) = y.v();
}
void backward_stack(Device *dev, Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d().slice(indexes(0, 0), indexes(nx, bs));
  y.d += z.d().slice(indexes(nx, 0), indexes(ny, bs));
}

// stacking with delay

void forward_stack_delay(Device *dev, Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
  z.v().slice(indexes(0, 0), indexes(nx, bs)) = x.v();
  if (last >= 0)
    z.v().slice(indexes(nx, 0), indexes(ny, bs)) = y[last].v();
  else
    z.v().slice(indexes(nx, 0), indexes(ny, bs)).setZero();
}
void backward_stack_delay(Device *dev, Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d().slice(indexes(0, 0), indexes(nx, bs));
  if (last >= 0) y[last].d += z.d().slice(indexes(nx, 0), indexes(ny, bs));
}

// reverse sequences

void forward_reverse(Device *dev, Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) y[N - i - 1] = x[i];
}
void backward_reverse(Device *dev, Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) x[N - i - 1].d += y[i].d();
}

// combine the delayed gated state with the gated input

void forward_statemem(Device *dev, Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf) {
  state.v = ci.v() * gi.v();
  if (last >= 0) state.v += gf.v() * states[last].v();
}
void backward_statemem(Device *dev, Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf) {
  if (last >= 0) states[last].d += state.d() * gf.v();
  if (last >= 0) gf.d += state.d() * states[last].v();
  gi.d += state.d() * ci.v();
  ci.d += state.d() * gi.v();
}

// nonlinear gated output

void forward_nonlingate(Device *dev, Batch &out, Batch &state, Batch &go, Nonlin nl) {
  Float (*f)(Float) = nonlinearities[nl].nonlin;
  out.v = state.v().unaryExpr(f) * go.v();
}
void backward_nonlingate(Device *dev, Batch &out, Batch &state, Batch &go, Nonlin nl) {
  Float (*f)(Float) = nonlinearities[nl].nonlin;
  Float (*g)(Float) = nonlinearities[nl].xderiv;
  go.d += state.v().unaryExpr(f) * out.d();
  state.d += state.v().unaryExpr(g) * go.v() * out.d();
}

}

