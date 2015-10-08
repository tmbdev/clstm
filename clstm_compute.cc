#include "clstm_compute.h"

#define A array()

namespace ocropus{

#define DOT(M, V) ((M) * (V))
#define MATMUL(A, B) ((A) * (B))
#define MATMUL_TR(A, B) ((A).transpose() * (B))
#define MATMUL_RT(A, B) ((A) * (B).transpose())
#define EMUL(U, V) ((U).array() * (V).array()).matrix()
#define EMULV(U, V) ((U).array() * (V).array()).matrix()
#define TRANPOSE(U) ((U).transpose())
#define COL(A, b) (A).col(b)
#define MAPFUNC(M, F) ((M).unaryExpr(F))
#define SUMREDUCE(M) float(M.sum())
#define BLOCK(A, i, j, n, m) (A).block(i, j, n, m)
#define CBUTFIRST(M) BLOCK((M), 0, 1, (M).rows(), (M).cols() - 1)
#define CFIRST(M) COL(M, 0)
#define HOMDOT(A1, B) (DOT(CBUTFIRST(A1), B).colwise() + CFIRST(A1))


inline void ADDCOLS(Mat &m, Vec &v) {
  for (int i = 0; i < COLS(m); i++)
    for (int j = 0; j < ROWS(m); j++) m(j, i) += v(j);
}

void gradient_clip(Sequence &s, Float m) {
  if (m < 0) return;
  for (int t = 0; t < s.size(); t++) {
    s[t].d =
        MAPFUNC(s[t].d, [m](Float x) { return x > m ? m : x < -m ? -m : x; });
  }
}

void gradient_clip(Mat &d, Float m) {
  if (m < 0) return;
  d = MAPFUNC(d, [m](Float x) { return x > m ? m : x < -m ? -m : x; });
}

template <class F>
void forward_full1(Batch &y, Params &W1, Batch &x) {
  y = HOMDOT(W1, x);
  F::f(y);
}
template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x, Float gc) {
  Mat temp;
  temp = y.d;
  F::df(temp, y);
  x.d = MATMUL_TR(CBUTFIRST(W1), temp);
  int bs = y.cols();
  auto d_W = CBUTFIRST(W1.d);
  d_W += MATMUL_RT(temp, x);
  auto d_w = CFIRST(W1.d);
  for (int b = 0; b < bs; b++) d_w += COL(temp, b);
}
template void forward_full1<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<NoNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x, Float gc);

// compute non-linear full layers
template <class F>
void forward_full(Batch &y, Params &W, Batch &x) {
  y = nonlin<F>(MATMUL(W, x));
}
template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc) {
  Mat temp = EMUL(yprime<F>(y), y.d);
  gradient_clip(temp, gc);
  x.d += MATMUL_TR(W, temp);
  W.d += MATMUL_RT(temp, x);
}
template void forward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<NoNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full<TanhNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full<ReluNonlin>(Batch &y, Params &W, Batch &x, Float gc);

void forward_softmax(Batch &z, Params &W1, Batch &x) {
  z = MAPFUN(HOMDOT(W1, x), limexp);
  for (int b = 0; b < COLS(z); b++) {
    Float total = fmax(SUMREDUCE(COL(z, b)), 1e-9);
    COL(z, b) /= total;
  }
}

void backward_softmax(Batch &z, Params &W1, Batch &x) {
  x.d = MATMUL_TR(CBUTFIRST(W1), z.d);
  auto d_W = CBUTFIRST(W1.d);
  d_W += MATMUL_RT(z.d, x);
  auto d_w = CFIRST(W1.d);
  int bs = COLS(z);
  for (int b = 0; b < bs; b++) d_w += COL(z.d, b);
}

void forward_stack(Batch &z, Batch &x, Batch &y) {
  assert(x.cols()==y.cols());
  int nx = x.rows();
  int ny = y.rows();
  int bs = x.cols();
  z.resize(nx+ny, bs);
  BLOCK(z, 0, 0, nx, bs) = x;
  BLOCK(z, nx, 0, ny, bs) = y;
}
void backward_stack(Batch &z, Batch &x, Batch &y) {
  assert(x.cols()==y.cols());
  int nx = x.rows();
  int ny = y.rows();
  int bs = x.cols();
  z.resize(nx+ny, bs);
  x.d += BLOCK(z.d, 0, 0, nx, bs);
  y.d += BLOCK(z.d, nx, 0, ny, bs);
}

void forward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  y.resize(N, x.rows(), x.cols());
  for (int i=0; i<N; i++) y[N-i-1] = x[i];
}
void backward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i=0; i<N; i++) x[N-i-1].d += y[i].d;
}

// stack the delayed output on the input
void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last) {
  assert(inp.cols() == out.cols());
  int bs = inp.cols();
  int ni = inp.rows();
  int no = out.rows();
  int nf = ni + no + 1;
  all.resize(nf, bs);
  BLOCK(all, 0, 0, 1, bs).setConstant(1);
  BLOCK(all, 1, 0, ni, bs) = inp;
  if (last < 0)
    BLOCK(all, 1 + ni, 0, no, bs).setConstant(0);
  else
    BLOCK(all, 1 + ni, 0, no, bs) = out[last];
}
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last) {
  assert(inp.cols() == out.cols());
  int bs = inp.cols();
  int ni = inp.rows();
  int no = out.rows();
  int nf = ni + no + 1;
  inp.d += BLOCK(all.d, 1, 0, ni, bs);
  if (last >= 0) out[last].d += BLOCK(all.d, 1 + ni, 0, no, bs);
}

// combine the delayed gated state with the gated input
void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf) {
  state = EMUL(ci, gi);
  if (last >= 0) state += EMUL(gf, states[last]);
}
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf) {
  if (last >= 0) states[last].d.A += state.d.A * gf.A;
  if (last >= 0) gf.d.A += state.d.A * states[last].A;
  gi.d.A += state.d.A * ci.A;
  ci.d.A += state.d.A * gi.A;
}

// nonlinear gated output
template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go) {
  out = EMUL(nonlin<H>(state), go);
}
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
  go.d.A += nonlin<H>(state).A * out.d.A;
  state.d.A += xprime<H>(state).A * go.A * out.d.A;
}

template void forward_nonlingate<TanhNonlin>(Batch &out, Batch &state, Batch &go);
template void forward_nonlingate<SigmoidNonlin>(Batch &out, Batch &state, Batch &go);
template void forward_nonlingate<NoNonlin>(Batch &out, Batch &state, Batch &go);
template void forward_nonlingate<ReluNonlin>(Batch &out, Batch &state, Batch &go);
template void backward_nonlingate<TanhNonlin>(Batch &out, Batch &state, Batch &go);
template void backward_nonlingate<SigmoidNonlin>(Batch &out, Batch &state, Batch &go);
template void backward_nonlingate<NoNonlin>(Batch &out, Batch &state, Batch &go);
template void backward_nonlingate<ReluNonlin>(Batch &out, Batch &state, Batch &go);

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

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

Vec timeslice(const Sequence &s, int i, int b) {
  Vec result(s.size());
  for (int t = 0; t < s.size(); t++) result[t] = s[t](i, b);
  return result;
}

bool anynan(Batch &a) {
  for (int j = 0; j < ROWS(a); j++) {
    for (int k = 0; k < COLS(a); k++) {
      float x = a(j, k);
      if (isnan(x)) return true;
    }
  }
}
bool anynan(Sequence &a) {
  for (int i = 0; i < a.size(); i++)
    if(anynan(a[i])) return true;
  return false;
}

}
