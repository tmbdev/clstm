#include "clstm_compute.h"
#include <iostream>

// FIXME: factor out nonlinearities

namespace ocropus {
using std::cerr;

#ifdef USEMAT
#define CBUTFIRST(M) (M).block(0, 1, (M).rows(), (M).cols() - 1)
#define CFIRST(M) (M).col(0)
#endif

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

// random initialization of sequences etc.


namespace {
double state = getenv("seed")?atof(getenv("seed")):0.1;

inline double randu() {
  state = 189843.9384938 * cos(state*193.3498);
  state -= floor(state);
  return state;
}
inline double randn() {
  double u1 = randu();
  double u2 = randu();
  double r = -2*log(u1);
  double theta = 2*M_PI*u2;
  double z0 = r * cos(theta);
  return z0;
}
}


void rinit(Ten2 m, Float s, const string mode, Float offset) {
  if (mode == "unif") {
    for(int i=0; i<rows(m); i++)
      for(int j=0;j<cols(m); j++)
        m(i,j) = 2 * s * randu() - s + offset;
  } else if (mode == "negbiased") {
    for(int i=0; i<rows(m); i++)
      for(int j=0;j<cols(m); j++)
        m(i,j) = 3 * s * randu() - 2 * s + offset;
  } else if (mode == "pos") {
    for(int i=0; i<rows(m); i++)
      for(int j=0;j<cols(m); j++)
        m(i,j) = s * randu() + offset;
  } else if (mode == "neg") {
    for(int i=0; i<rows(m); i++)
      for(int j=0;j<cols(m); j++)
        m(i,j) = - s * randu() + offset;
  } else if (mode == "normal") {
    for(int i=0; i<rows(m); i++)
      for(int j=0;j<cols(m); j++)
        m(i,j) = s * randn() + offset;
  }
}

void rinit(Params &m, int r, int c, Float s, const string mode, Float offset) {
  m.resize(r,c);
  rinit(m.V(), s, mode, offset);
}
void rinit(Batch &m, int r, int c, Float s, const string mode, Float offset) {
  m.resize(r,c);
  rinit(m.V(), s, mode, offset);
}
void rinit(Sequence &m, int N, int r, int c, Float s, const string mode, Float offset) {
  m.resize(N,r,c);
  for(int t=0; t<N; t++)
    rinit(m[t].V(), s, mode, offset);
}


// checking for NaNs in different objects

bool anynan(Ten2 a) {
  for (int j = 0; j < rows(a); j++) {
    for (int k = 0; k < cols(a); k++) {
      float x = a(j, k);
      if (isnan(x)) return true;
    }
  }
}

bool anynan(Batch &a) { return anynan(a.V()) || anynan(a.D()); }

bool anynan(Sequence &a) {
  for (int i = 0; i < a.size(); i++)
    if (anynan(a[i])) return true;
  return false;
}

// helper functions for Eigen::Tensor axes and sizes

inline array<Eigen::IndexPair<int>, 1> axes(int i, int j) {
  array<Eigen::IndexPair<int>, 1> result = {Eigen::IndexPair<int>(i, j)};
  return result;
}

inline array<ptrdiff_t, 1> ar(int i) { return array<ptrdiff_t, 1>({i}); }

inline array<ptrdiff_t, 2> ar(int i, int j) {
  return array<ptrdiff_t, 2>({i, j});
}

inline Eigen::Sizes<1> S(int i) { return Eigen::Sizes<1>({i}); }

inline Eigen::Sizes<2> S(int i, int j) { return Eigen::Sizes<2>({i, j}); }

// full layers with constant offset

template <class F>
void forward_full1(Batch &y, Params &W1, Batch &x) {
  Float (*f)(Float) = F::nonlin;
#ifndef USEMAT
  int n = W1.V().dimension(0), m = W1.V().dimension(1);
  int bs = x.V().dimension(1);
  y.V() =
      (W1.V().slice(ar(0, 1), ar(n, m - 1)).contract(x.V(), axes(1, 0)) +
       W1.V().chip(0, 1).reshape(ar(n, 1)).broadcast(ar(1, bs))).unaryExpr(f);
#else
  y.v = (CBUTFIRST(W1.v) * x.v).colwise() + CFIRST(W1.v);
  y.v = y.v.unaryExpr(f);
#endif
}
template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x) {
  Float (*g)(Float) = F::yderiv;
#ifndef USEMAT
  int n = W1.V().dimension(0), m = W1.V().dimension(1);
  Tensor2 temp = y.D() * y.V().unaryExpr(g);
  x.D() = W1.V().slice(ar(0, 1), ar(n, m - 1)).contract(temp, axes(0, 0));
  W1.D().slice(ar(0, 1), ar(n, m - 1)) += temp.contract(x.V(), axes(1, 1));
  W1.D().chip(0, 1) += temp.sum(ar(1));
#else
  Mat temp;
  temp.array() = y.d.array() * y.v.array().unaryExpr(g);
  x.d = CBUTFIRST(W1.v).transpose() * temp;
  int bs = y.v.cols();
  auto d_W = CBUTFIRST(W1.d);
  d_W += temp * x.v.transpose();
  auto d_w = CFIRST(W1.d);
  for (int b = 0; b < bs; b++) d_w += temp.col(b);
#endif
}
template void forward_full1<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<NoNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x);

// full layers without constant offset

template <class F>
void forward_full(Batch &y, Params &W, Batch &x) {
  Float (*f)(Float) = F::nonlin;
#ifndef USEMAT
  y.V() = W.V().contract(x.V(), axes(1, 0)).unaryExpr(f);
#else
  y.v = (W.v * x.v).unaryExpr(f);;
#endif
}
template <class F>
void backward_full(Batch &y, Params &W, Batch &x) {
  Float (*g)(Float) = F::yderiv;
#ifndef USEMAT
  Tensor2 temp = y.V().unaryExpr(g) * y.D();
  x.D() += W.V().contract(temp, axes(0, 0));
  W.D() += temp.contract(x.V(), axes(1, 1));
#else
  Mat temp = y.d.array() * y.v.unaryExpr(g).array();
  x.d += W.v.transpose() * temp;
  W.d += temp * x.v.transpose();
#endif
}
template void forward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);

// softmax

void forward_softmax(Batch &z, Params &W1, Batch &x) {
  Float (*f)(Float) = limexp;
#ifndef USEMAT
  int n = W1.V().dimension(0);
  int m = W1.V().dimension(1);
  int bs = z.V().dimension(1);
  z.V() =
      (W1.V().slice(ar(0, 1), ar(n, m - 1)).contract(x.V(), axes(1, 0)) +
       W1.V().chip(0, 1).reshape(ar(n, 1)).broadcast(ar(1, bs))).unaryExpr(f);
  for (int b = 0; b < bs; b++) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += z.V()(i, b);
    for (int i = 0; i < n; i++) z.V()(i, b) /= total;
  }
#else
  int n = ROWS(W1.v);
  int m = COLS(W1.v);
  int bs = COLS(x.v);
  z.v = ((CBUTFIRST(W1.v) * x.v).colwise() + CFIRST(W1.v)).unaryExpr(f);
  for (int b = 0; b < bs; b++) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += z.v(i, b);
    for (int i = 0; i < n; i++) z.v(i, b) /= total;
  }
#endif
}
void backward_softmax(Batch &z, Params &W1, Batch &x) {
#ifndef USEMAT
  int n = W1.V().dimension(0), m = W1.V().dimension(1);
  int bs = z.V().dimension(1);
  x.D() = W1.V().slice(ar(0, 1), ar(n, m - 1)).contract(z.D(), axes(0, 0));
  W1.D().slice(ar(0, 1), ar(n, m - 1)) += z.D().contract(x.V(), axes(1, 1));
  for (int i = 0; i < n; i++)
    for (int b = 0; b < bs; b++) W1.D()(i, 0) += z.D()(i, b);
#else
  x.d = CBUTFIRST(W1.v).transpose() * z.d;
  auto d_W = CBUTFIRST(W1.d);
  d_W += z.d * x.v.transpose();
  int n = ROWS(W1.v);
  int bs = COLS(z.v);
  Vec d_w = CFIRST(W1.d);
  for (int i = 0; i < n; i++)
    for (int b = 0; b < bs; b++) d_w(i) += z.d(i, b);
  CFIRST(W1.d) = d_w;
#endif
}

// stacking

void forward_stack(Batch &z, Batch &x, Batch &y) {
#ifndef USEMAT
  int nx = x.V().dimension(0), ny = y.V().dimension(0);
  int bs = x.V().dimension(1);
  z.V().slice(ar(0, 0), ar(nx, bs)) = x.V();
  z.V().slice(ar(nx, 0), ar(ny, bs)) = y.V();
#else
  assert(x.cols() == y.cols());
  int nx = x.rows();
  int ny = y.rows();
  int bs = x.cols();
  z.v.block(0, 0, nx, bs) = x.v;
  z.v.block(nx, 0, ny, bs) = y.v;
#endif
}
void backward_stack(Batch &z, Batch &x, Batch &y) {
#ifndef USEMAT
  int nx = x.V().dimension(0), ny = y.V().dimension(0);
  int bs = x.V().dimension(1);
  x.D() += z.D().slice(ar(0, 0), ar(nx, bs));
  y.D() += z.D().slice(ar(nx, 0), ar(ny, bs));
#else
  assert(x.cols() == y.cols());
  int nx = x.rows();
  int ny = y.rows();
  int bs = x.cols();
  x.d += z.d.block(0, 0, nx, bs);
  y.d += z.d.block(nx, 0, ny, bs);
#endif
}

// stacking with delay

void forward_stack(Batch &z, Batch &x, Sequence &y, int last) {
#ifndef USEMAT
  int nx = x.V().dimension(0), ny = y[0].V().dimension(0);
  int bs = x.V().dimension(1);
  z.V().slice(ar(0, 0), ar(nx, bs)) = x.V();
  if (last >= 0)
    z.V().slice(ar(nx, 0), ar(ny, bs)) = y[last].V();
  else
    z.V().slice(ar(nx, 0), ar(ny, bs)).setZero();
#else
  assert(x.cols() == y.cols());
  int nx = x.rows();
  int ny = y.rows();
  int bs = x.cols();
  z.v.block(0, 0, nx, bs) = x.v;
  if (last >= 0)
    z.v.block(nx, 0, ny, bs) = y[last].v;
  else
    z.v.block(nx, 0, ny, bs).setZero();
#endif
}
void backward_stack(Batch &z, Batch &x, Sequence &y, int last) {
#ifndef USEMAT
  int nx = x.V().dimension(0), ny = y[0].V().dimension(0);
  int bs = x.V().dimension(1);
  x.D() += z.D().slice(ar(0, 0), ar(nx, bs));
  if (last >= 0) y[last].D() += z.D().slice(ar(nx, 0), ar(ny, bs));
#else
  assert(x.cols() == y.cols());
  int nx = x.rows();
  int ny = y.rows();
  int bs = x.cols();
  x.d += z.d.block(0, 0, nx, bs);
  if (last >= 0) y[last].d += z.d.block(nx, 0, ny, bs);
#endif
}

// stacking with delay and adding a constant

void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last) {
#ifndef USEMAT
  int nx = inp.V().dimension(0), ny = out[0].V().dimension(0);
  int bs = inp.V().dimension(1);
  all.V().slice(ar(0, 0), ar(1, bs)).setConstant(Float(1));
  all.V().slice(ar(1, 0), ar(nx, bs)) = inp.V();
  if (last >= 0)
    all.V().slice(ar(1 + nx, 0), ar(ny, bs)) = out[last].V();
  else
    all.V().slice(ar(1 + nx, 0), ar(ny, bs)).setZero();
#else
  assert(inp.cols() == out.cols());
  int bs = inp.cols();
  int ni = inp.rows();
  int no = out.rows();
  int nf = ni + no + 1;
  all.v.block(0, 0, 1, bs).setConstant(1);
  all.v.block(1, 0, ni, bs) = inp.v;
  if (last < 0)
    all.v.block(1 + ni, 0, no, bs).setConstant(0);
  else
    all.v.block(1 + ni, 0, no, bs) = out[last].v;
#endif
}
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last) {
#ifndef USEMAT
  int nx = inp.V().dimension(0), ny = out[0].V().dimension(0);
  int bs = inp.V().dimension(1);
  inp.D() += all.D().slice(ar(1, 0), ar(nx, bs));
  if (last >= 0) out[last].D() += all.D().slice(ar(1 + nx, 0), ar(ny, bs));
#else
  assert(inp.cols() == out.cols());
  int bs = inp.cols();
  int ni = inp.rows();
  int no = out.rows();
  int nf = ni + no + 1;
  inp.d += all.d.block(1, 0, ni, bs);
  if (last >= 0) out[last].d += all.d.block(1 + ni, 0, no, bs);
#endif
}

// reverse sequences

void forward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) y[N - i - 1] = x[i];
}
void backward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) x[N - i - 1].D() += y[i].D();
}

// combine the delayed gated state with the gated input

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf) {
#ifndef USEMAT
  state.V() = ci.V() * gi.V();
  if (last >= 0) state.V() += gf.V() * states[last].V();
#else
  state.v = ci.v.array() * gi.v.array();
  if (last >= 0) state.v.array() += gf.v.array() * states[last].v.array();
#endif
}
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf) {
#ifndef USEMAT
  if (last >= 0) states[last].D() += state.D() * gf.V();
  if (last >= 0) gf.D() += state.D() * states[last].V();
  gi.D() += state.D() * ci.V();
  ci.D() += state.D() * gi.V();
#else
  if (last >= 0) states[last].d.array() += state.d.array() * gf.v.array();
  if (last >= 0) gf.d.array() += state.d.array() * states[last].v.array();
  gi.d.array() += state.d.array() * ci.v.array();
  ci.d.array() += state.d.array() * gi.v.array();
#endif
}

// nonlinear gated output

template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
#ifndef USEMAT
  out.V() = state.V().unaryExpr(f) * go.V();
#else
  out.v = state.v.unaryExpr(f).array() * go.v.array();
#endif
}
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
  auto g = [](Float x) { return H::yderiv(H::nonlin(x)); };
#ifndef USEMAT
  go.D() += state.V().unaryExpr(f) * out.D();
  state.D() += state.V().unaryExpr(g) * go.V() * out.D();
#else
  go.d.array() += state.v.unaryExpr(f).array() * out.d.array();
  state.d.array() += state.v.unaryExpr(g).array() * go.v.array() * out.d.array();
#endif
}

template void forward_nonlingate<TanhNonlin>(Batch &out, Batch &state,
                                             Batch &go);
template void forward_nonlingate<SigmoidNonlin>(Batch &out, Batch &state,
                                                Batch &go);
template void forward_nonlingate<NoNonlin>(Batch &out, Batch &state, Batch &go);
template void forward_nonlingate<ReluNonlin>(Batch &out, Batch &state,
                                             Batch &go);
template void backward_nonlingate<TanhNonlin>(Batch &out, Batch &state,
                                              Batch &go);
template void backward_nonlingate<SigmoidNonlin>(Batch &out, Batch &state,
                                                 Batch &go);
template void backward_nonlingate<NoNonlin>(Batch &out, Batch &state,
                                            Batch &go);
template void backward_nonlingate<ReluNonlin>(Batch &out, Batch &state,
                                              Batch &go);
}
