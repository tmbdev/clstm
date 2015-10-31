#include "clstm_compute.h"
#include <iostream>

// FIXME: factor out nonlinearities

namespace ocropus {
using std::cerr;

inline Eigen::array<Eigen::IndexPair<int>, 1> axes(int i, int j) {
  Eigen::array<Eigen::IndexPair<int>, 1> result = {Eigen::IndexPair<int>(i, j)};
  return result;
}

inline Eigen::array<ptrdiff_t, 1> indexes(int i) { 
  return Eigen::array<ptrdiff_t, 1>({i}); 
}

inline Eigen::array<ptrdiff_t, 2> indexes(int i, int j) {
  return Eigen::array<ptrdiff_t, 2>({i, j});
}

#ifdef USEMAT
#define CBUTFIRST(M) (M).block(0, 1, (M).rows(), (M).cols() - 1)
#define CFIRST(M) (M).col(0)
#endif

// unordered_map<Float*,int> refcounts;
// shared_ptr<Context> default_context(new ThreadPoolContext(4));

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

void rinit(TensorMap2 m, Float s, const string mode, Float offset) {
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
  rinit(m.v(), s, mode, offset);
}
void rinit(Batch &m, int r, int c, Float s, const string mode, Float offset) {
  m.resize(r,c);
  rinit(m.v(), s, mode, offset);
}
void rinit(Sequence &m, int N, int r, int c, Float s, const string mode, Float offset) {
  m.resize(N,r,c);
  for(int t=0; t<N; t++)
    rinit(m[t].v(), s, mode, offset);
}


// checking for NaNs in different objects

bool anynan(TensorMap2 a) {
  for (int j = 0; j < rows(a); j++) {
    for (int k = 0; k < cols(a); k++) {
      float x = a(j, k);
      if (isnan(x)) return true;
    }
  }
  return false;
}

bool anynan(Batch &a) { return anynan(a.v()) || anynan(a.d()); }

bool anynan(Sequence &a) {
  for (int i = 0; i < a.size(); i++)
    if (anynan(a[i])) return true;
  return false;
}

// full layers with constant offset

template <class F>
void forward_full1(Batch &y, Params &W1, Batch &x) {
  Float (*f)(Float) = F::nonlin;
#ifndef USEMAT
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = x.v.dimension(1);
  y.v =
      (W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(x.v(), axes(1, 0)) +
       W1.v().chip(0, 1).reshape(indexes(n, 1)).broadcast(indexes(1, bs))).unaryExpr(f);
#else
  y.v = (CBUTFIRST(W1.v) * x.v).colwise() + CFIRST(W1.v);
  y.v = y.v.unaryExpr(f);
#endif
}
template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x) {
  Float (*g)(Float) = F::yderiv;
#ifndef USEMAT
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  EigenTensor2 temp = y.d() * y.v().unaryExpr(g);
  x.d = W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(temp, axes(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)) += temp.contract(x.v(), axes(1, 1));
  W1.d().chip(0, 1) += temp.sum(indexes(1));
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
  y.v = W.v().contract(x.v(), axes(1, 0)).unaryExpr(f);
#else
  y.v = (W.v * x.v).unaryExpr(f);;
#endif
}
template <class F>
void backward_full(Batch &y, Params &W, Batch &x) {
  Float (*g)(Float) = F::yderiv;
#ifndef USEMAT
  EigenTensor2 temp = y.v().unaryExpr(g) * y.d();
  x.d += W.v().contract(temp, axes(0, 0));
  W.d += temp.contract(x.v(), axes(1, 1));
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
  int n = W1.v.dimension(0);
  int m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  z.v =
      (W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(x.v(), axes(1, 0)) +
       W1.v().chip(0, 1).reshape(indexes(n, 1)).broadcast(indexes(1, bs))).unaryExpr(f);
  for (int b = 0; b < bs; b++) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += z.v()(i, b);
    for (int i = 0; i < n; i++) z.v(i, b) /= total;
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
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  x.d = W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(z.d(), axes(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)) += z.d().contract(x.v(), axes(1, 1));
  for (int i = 0; i < n; i++)
    for (int b = 0; b < bs; b++) W1.d(i, 0) += z.d()(i, b);
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
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  z.v().slice(indexes(0, 0), indexes(nx, bs)) = x.v();
  z.v().slice(indexes(nx, 0), indexes(ny, bs)) = y.v();
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
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d().slice(indexes(0, 0), indexes(nx, bs));
  y.d += z.d().slice(indexes(nx, 0), indexes(ny, bs));
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
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  z.v().slice(indexes(0, 0), indexes(nx, bs)) = x.v();
  if (last >= 0)
    z.v().slice(indexes(nx, 0), indexes(ny, bs)) = y[last].v();
  else
    z.v().slice(indexes(nx, 0), indexes(ny, bs)).setZero();
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
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d().slice(indexes(0, 0), indexes(nx, bs));
  if (last >= 0) y[last].d += z.d().slice(indexes(nx, 0), indexes(ny, bs));
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
  int nx = inp.v.dimension(0), ny = out[0].v.dimension(0);
  int bs = inp.v.dimension(1);
  all.v().slice(indexes(0, 0), indexes(1, bs)).setConstant(Float(1));
  all.v().slice(indexes(1, 0), indexes(nx, bs)) = inp.v();
  if (last >= 0)
    all.v().slice(indexes(1 + nx, 0), indexes(ny, bs)) = out[last].v();
  else
    all.v().slice(indexes(1 + nx, 0), indexes(ny, bs)).setZero();
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
  int nx = inp.v.dimension(0), ny = out[0].v.dimension(0);
  int bs = inp.v.dimension(1);
  inp.d += all.d().slice(indexes(1, 0), indexes(nx, bs));
  if (last >= 0) out[last].d += all.d().slice(indexes(1 + nx, 0), indexes(ny, bs));
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
  for (int i = 0; i < N; i++) x[N - i - 1].d += y[i].d();
}

// combine the delayed gated state with the gated input

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf) {
#ifndef USEMAT
  state.v = ci.v() * gi.v();
  if (last >= 0) state.v += gf.v() * states[last].v();
#else
  state.v = ci.v.array() * gi.v.array();
  if (last >= 0) state.v.array() += gf.v.array() * states[last].v.array();
#endif
}
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf) {
#ifndef USEMAT
  if (last >= 0) states[last].d += state.d() * gf.v();
  if (last >= 0) gf.d += state.d() * states[last].v();
  gi.d += state.d() * ci.v();
  ci.d += state.d() * gi.v();
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
  out.v = state.v().unaryExpr(f) * go.v();
#else
  out.v = state.v.unaryExpr(f).array() * go.v.array();
#endif
}
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
  auto g = [](Float x) { return H::yderiv(H::nonlin(x)); };
#ifndef USEMAT
  go.d += state.v().unaryExpr(f) * out.d();
  state.d += state.v().unaryExpr(g) * go.v() * out.d();
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
