#include "clstm_compute.h"

#define A array()

namespace ocropus{
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
  y = MATMUL(CBUTFIRST(W1), x).colwise() + CFIRST(W1);
  F::f(y);
}
template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x, Float gc) {
  int bs = y.cols();
  auto W = CBUTFIRST(W1);
  auto w = CFIRST(W1);
  auto d_W = CBUTFIRST(W1.d);
  auto d_w = CBUTFIRST(W1.d);
  Mat temp = EMUL(yprime<F>(y), y.d);
  gradient_clip(temp, gc);
  x.d += MATMUL_TR(W, temp);
  d_W += MATMUL_RT(temp, x);
  for (int b = 0; b < bs; b++) d_w += COL(y.d, b);
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
  x.d = BLOCK(z.d, 0, 0, nx, bs);
  y.d = BLOCK(z.d, nx, 0, ny, bs);
}

void forward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  y.resize(N, x.rows(), x.cols());
  for (int i=0; i<N; i++) y[N-i-1] = x[i];
}
void backward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i=0; i<N; i++) x[N-i-1].d = y[i].d;
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

}
