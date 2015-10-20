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

void randgauss(Mat &m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> randn;
  for (int i = 0; i < ROWS(m); i++)
    for (int j = 0; j < COLS(m); j++) m(i, j) = randn(gen);
}

void randgauss(Vec &v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> randn;
  for (int i = 0; i < ROWS(v); i++) v(i) = randn(gen);
}

void randinit(Mat &m, float s, const string mode) {
  if (mode == "unif") {
    m.setRandom();
    m = (2 * s * m).array() - s;
  } else if (mode == "pos") {
    m.setRandom();
    m = m * s;
  } else if (mode == "normal") {
    randgauss(m);
    m = m * s;
  }
}

void randinit(Vec &m, float s, const string mode) {
  if (mode == "unif") {
    m.setRandom();
    m = (2 * s * m).array() - s;
  } else if (mode == "pos") {
    m.setRandom();
    m = m * s;
  } else if (mode == "normal") {
    randgauss(m);
    m = m * s;
  }
}
void randinit(Batch &m, int no, int ni, float s, const string mode) {
  m.resize(no, ni);
  randinit(m.v, s, mode);
}
void randinit(Mat &m, int no, int ni, float s, const string mode) {
  m.resize(no, ni);
  randinit(m, s, mode);
}
void randinit(Vec &m, int no, float s, const string mode) {
  m.resize(no);
  randinit(m, s, mode);
}
void zeroinit(Mat &m, int no, int ni) {
  m.resize(no, ni);
  m.setZero();
}
void zeroinit(Vec &m, int no) {
  m.resize(no);
  m.setZero();
}

void resize(Sequence &seq, int nsteps, int dims, int bs) {
  seq.resize(nsteps);
  for (int i = 0; i < nsteps; i++) seq[i].resize(dims, bs);
}

int size(Sequence &seq, int dim) {
  if (dim == 0) return seq.size();
  if (dim == 1) return seq[0].rows();
  if (dim == 2) return seq[0].cols();
  THROW("bad dim ins size");
  return -1;
}

Vec timeslice(const Sequence &s, int i, int b) {
  Vec result(s.size());
  for (int t = 0; t < s.size(); t++) result[t] = s[t].v(i, b);
  return result;
}

// checking for NaNs in different objects

bool anynan(Mat &a) {
  for (int j = 0; j < ROWS(a); j++) {
    for (int k = 0; k < COLS(a); k++) {
      float x = a(j, k);
      if (isnan(x)) return true;
    }
  }
}

bool anynan(Batch &a) { return anynan(a.v) || anynan(a.d); }

bool anynan(Sequence &a) {
  for (int i = 0; i < a.size(); i++)
    if (anynan(a[i])) return true;
  return false;
}

// clipping

void gradient_clip(Sequence &s, Float m) {
  if (m < 0) return;
  for (int t = 0; t < s.size(); t++) {
    s[t].d =
        s[t].d.unaryExpr([m](Float x) { return x > m ? m : x < -m ? -m : x; });
  }
}

void gradient_clip(Mat &d, Float m) {
  if (m < 0) return;
  d = d.unaryExpr([m](Float x) { return x > m ? m : x < -m ? -m : x; });
}

void gradient_clip(Batch &b, Float m) { gradient_clip(b.d, m); }

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
#ifndef USEMAT
  int n = W1.V().dimension(0), m = W1.V().dimension(1);
  int bs = x.V().dimension(1);
  Float (*f)(Float) = F::nonlin;
  y.V() =
      (W1.V().slice(ar(0, 1), ar(n, m - 1)).contract(x.V(), axes(1, 0)) +
       W1.V().chip(0, 1).reshape(ar(n, 1)).broadcast(ar(1, bs))).unaryExpr(f);
#else
  y.v = (CBUTFIRST(W1.v) * x.v).colwise() + CFIRST(W1.v);
  F::f(y.v);
#endif
}
template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x, Float gc) {
#ifndef USEMAT
  int n = W1.V().dimension(0), m = W1.V().dimension(1);
  Float (*g)(Float) = F::yderiv;
  Tensor2 temp = y.D() * y.V().unaryExpr(g);
  x.D() = W1.V().slice(ar(0, 1), ar(n, m - 1)).contract(temp, axes(0, 0));
  W1.D().slice(ar(0, 1), ar(n, m - 1)) += temp.contract(x.V(), axes(1, 1));
  W1.D().chip(0, 1) += temp.sum(ar(1));
#else
  Mat temp;
  temp = y.d;
  F::df(temp, y.v);
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
template void backward_full1<NoNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x,
                                            Float gc);
template void backward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x,
                                         Float gc);
template void backward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x,
                                         Float gc);

// full layers without constant offset

template <class F>
void forward_full(Batch &y, Params &W, Batch &x) {
#ifndef USEMAT
  Float (*f)(Float) = F::nonlin;
  y.V() = W.V().contract(x.V(), axes(1, 0)).unaryExpr(f);
#else
  y.v = nonlin<F>(W.v * x.v);
#endif
}
template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc) {
#ifndef USEMAT
  Float (*g)(Float) = F::yderiv;
  Tensor2 temp = y.V().unaryExpr(g) * y.D();
  x.D() += W.V().contract(temp, axes(0, 0));
  W.D() += temp.contract(x.V(), axes(1, 1));
#else
  Mat temp = yprime<F>(y.v).array() * y.d.array();
  gradient_clip(temp, gc);
  x.d += W.v.transpose() * temp;
  W.d += temp * x.v.transpose();
#endif
}
template void forward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<NoNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x,
                                           Float gc);
template void backward_full<TanhNonlin>(Batch &y, Params &W, Batch &x,
                                        Float gc);
template void backward_full<ReluNonlin>(Batch &y, Params &W, Batch &x,
                                        Float gc);

// softmax

void forward_softmax(Batch &z, Params &W1, Batch &x) {
#ifndef USEMAT
  int n = W1.V().dimension(0);
  int m = W1.V().dimension(1);
  int bs = z.V().dimension(1);
  Float (*f)(Float) = limexp;
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
  Float (*f)(Float) = limexp;
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
  for (int i = 0; i < N; i++) x[N - i - 1].d += y[i].d;
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
#ifndef USEMAT
  Float (*f)(Float) = H::nonlin;
  out.V() = state.V().unaryExpr(f) * go.V();
#else
  out.v = nonlin<H>(state.v).array() * go.v.array();
#endif
}
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
#ifndef USEMAT
  Float (*f)(Float) = H::nonlin;
  auto g = [](Float x) { return H::yderiv(H::nonlin(x)); };
  go.D() += state.V().unaryExpr(f) * out.D();
  state.D() += state.V().unaryExpr(g) * go.V() * out.D();
#else
  go.d.array() += nonlin<H>(state.v).array() * out.d.array();
  state.d.array() += xprime<H>(state.v).array() * go.v.array() * out.d.array();
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
