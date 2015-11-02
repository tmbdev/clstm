#include "clstm_compute.h"
#include <iostream>

// FIXME: factor out nonlinearities
// FIXME: remove _full(...) calls

namespace ocropus {
using std::cerr;

Context default_context;

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
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = x.v.dimension(1);
  assert(y.rows() == n);
  assert(y.cols() == x.cols());
  assert(x.rows() == m-1);
  y.v =
      (W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(x.v(), axispairs(1, 0)) +
       W1.v().chip(0, 1).reshape(indexes(n, 1)).broadcast(indexes(1, bs))).unaryExpr(f);
}
template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x) {
  Float (*g)(Float) = F::yderiv;
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  EigenTensor2 temp = y.v().unaryExpr(g) * y.d();
  x.d += W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(temp, axispairs(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)) += temp.contract(x.v(), axispairs(1, 1));
  W1.d().chip(0, 1) += temp.sum(indexes(1));
}
template void forward_full1<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<NoNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x);

// softmax

void forward_softmax(Batch &z, Params &W1, Batch &x) {
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
#if 1
  EigenTensor1 sums = z.v().sum(indexes(0));
  assert(sums.dimension(0)==bs);
  z.v = z.v() / sums.reshape(indexes(1,bs)).broadcast(indexes(n,1));;
#else
  TensorMap2 v = z.v();
  for (int b = 0; b < bs; b++) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += v(i, b);
    for (int i = 0; i < n; i++) v(i, b) /= total;
  }
#endif
}
void backward_softmax(Batch &z, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = z.v.dimension(1);
  x.d = W1.v().slice(indexes(0, 1), indexes(n, m - 1)).contract(z.d(), axispairs(0, 0));
  W1.d().slice(indexes(0, 1), indexes(n, m - 1)) += z.d().contract(x.v(), axispairs(1, 1));
  W1.d().chip(0, 1) += z.d().sum(indexes(1));
}

// stacking

void forward_stack(Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  assert(z.rows() == x.rows() + y.rows());
  assert(z.cols() == x.cols() && z.cols() == y.cols());
  z.v().slice(indexes(0, 0), indexes(nx, bs)) = x.v();
  z.v().slice(indexes(nx, 0), indexes(ny, bs)) = y.v();
}
void backward_stack(Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d().slice(indexes(0, 0), indexes(nx, bs));
  y.d += z.d().slice(indexes(nx, 0), indexes(ny, bs));
}

// stacking with delay

void forward_stack(Batch &z, Batch &x, Sequence &y, int last) {
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
void backward_stack(Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d().slice(indexes(0, 0), indexes(nx, bs));
  if (last >= 0) y[last].d += z.d().slice(indexes(nx, 0), indexes(ny, bs));
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
  state.v = ci.v() * gi.v();
  if (last >= 0) state.v += gf.v() * states[last].v();
}
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf) {
  if (last >= 0) states[last].d += state.d() * gf.v();
  if (last >= 0) gf.d += state.d() * states[last].v();
  gi.d += state.d() * ci.v();
  ci.d += state.d() * gi.v();
}

// nonlinear gated output

template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
  out.v = state.v().unaryExpr(f) * go.v();
}
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
  auto g = [](Float x) { return H::yderiv(H::nonlin(x)); };
  go.d += state.v().unaryExpr(f) * out.d();
  state.d += state.v().unaryExpr(g) * go.v() * out.d();
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

#ifdef DEPRECATED
// full layers without constant offset

template <class F>
void forward_full(Batch &y, Params &W, Batch &x) {
  assert(y.rows() == W.rows());
  assert(y.cols() == x.cols());
  assert(x.rows() == W.cols());
  Float (*f)(Float) = F::nonlin;
  y.v = W.v().contract(x.v(), axispairs(1, 0)).unaryExpr(f);
}
template <class F>
void backward_full(Batch &y, Params &W, Batch &x) {
  Float (*g)(Float) = F::yderiv;
  EigenTensor2 temp = y.v().unaryExpr(g) * y.d();
  x.d += W.v().contract(temp, axispairs(0, 0));
  W.d += temp.contract(x.v(), axispairs(1, 1));
}
template void forward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);

// stacking with delay and adding a constant

void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last) {
  int nx = inp.v.dimension(0), ny = out[0].v.dimension(0);
  int bs = inp.v.dimension(1);
  assert(all.rows() == 1 + inp.rows() + out.rows());
  assert(all.cols() == inp.cols() && all.cols() == out.cols());
  all.v().slice(indexes(0, 0), indexes(1, bs)).setConstant(Float(1));
  all.v().slice(indexes(1, 0), indexes(nx, bs)) = inp.v();
  if (last >= 0)
    all.v().slice(indexes(1 + nx, 0), indexes(ny, bs)) = out[last].v();
  else
    all.v().slice(indexes(1 + nx, 0), indexes(ny, bs)).setZero();
}
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last) {
  int nx = inp.v.dimension(0), ny = out[0].v.dimension(0);
  int bs = inp.v.dimension(1);
  inp.d += all.d().slice(indexes(1, 0), indexes(nx, bs));
  if (last >= 0) out[last].d += all.d().slice(indexes(1 + nx, 0), indexes(ny, bs));
}
#endif

