#ifndef clstm_compute__
#define clstm_compute__

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "tensor.h"

namespace ocropus {
using namespace std;

inline Float tanh_(Float x) { return tanh(x); }
inline Float relu_(Float x) { return x <= 0 ? 0 : x; }
inline Float heavi_(Float x) { return x <= 0 ? 0 : 1; }

#ifndef MAXEXP
#define MAXEXP 30
#endif

inline Float limexp(Float x) {
#if 1
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
#else
  return exp(x);
#endif
}

inline Float sigmoid(Float x) {
#if 1
  return 1.0 / (1.0 + limexp(-x));
#else
  return 1.0 / (1.0 + exp(-x));
#endif
}

inline Float log_add(Float x, Float y) {
  if (abs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline Float log_mul(Float x, Float y) { return x + y; }

struct Batch {
  tensor2 v;
  tensor2 d;
  Ten2 V() { return Ten2(v.data(), v.rows(), v.cols()); }
  Ten2 D() { return Ten2(d.data(), d.rows(), d.cols()); }
#ifdef USEMAT
  Mat &MV() { return *(Mat*)0; }
  Mat &MD() { return *(Mat*)0; }
#endif
  int rows() const { return v.dimension(0); }
  int cols() const { return v.dimension(1); }
  void setZero(int n, int m) {
    v.setZero(n, m);
    d.setZero(n, m);
  }
  void resize(int n, int m) { setZero(n, m); }
  void clear() {
    v.setZero();
    d.setZero();
  }
  void zeroGrad() { d.setZero(rows(), cols()); }
  void gradientClip(Float clip) {
    Ten2 d = *this->d;
    if (clip>=1e6) return;
    assert(clip>0);
    for(int i=0; i<rows(); i++) {
      for(int j=0; j<cols(); j++) {
        d(i,j) = fmax(-clip, fmin(clip, d(i,j)));
      }
    }
  }
};
struct Params : Batch {
  void update(Float lr, Float mom) {
    Ten2 v = *this->v;
    Ten2 d = *this->d;
    v += d * lr;
    d = d * mom;
  }
};

// typedef vector<Mat> Sequence;
struct Sequence {
  vector<Batch> steps;
  Sequence() {}
  Sequence(int N, int r, int b) { resize(N, r, b); }
  void clear() { steps.clear(); }
  int rows() const { return steps[0].rows(); }
  int cols() const { return steps[0].cols(); }
  void check() const {
    int N = steps.size();
    if (N == 0) return;
    assert(steps[0].rows() > 0);
    assert(steps[0].cols() > 0);
    for (int t = 0; t < N; t++) {
      assert(steps[t].rows() == steps[0].rows());
      assert(steps[t].cols() == steps[0].cols());
    }
  }
  int size() const { return steps.size(); }
  void resize(int n) { resize(n, 1, 1); }
  void resize(int N, int n, int m) {
    steps.resize(N);
    for (int t = 0; t < N; t++) {
      steps[t].resize(n, m);
    }
  }
  void like(const Sequence &other) {
    resize(other.size(), other.rows(), other.cols());
  }
  void copy(const Sequence &other) {
    resize(other.size());
    for (int t = 0; t < other.size(); t++) steps[t] = other[t];
  }
  Batch &operator[](int i) { return steps[i]; }
  const Batch &operator[](int i) const { return steps[i]; }
  void zero() {
    for (int t = 0; t < steps.size(); t++) steps[t].clear();
  }
  void zeroGrad() {
    for (int t = 0; t < steps.size(); t++) steps[t].zeroGrad();
  }
  void gradientClip(Float gc) {
    for(int i=0;i<steps.size(); i++)
      steps[i].gradientClip(gc);
  }
};

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

struct NoNonlin {
  static constexpr const char *kind = "Linear";
  static inline Float nonlin(Float x) { return x; }
  static inline Float yderiv(Float y) { return 1; }
};
struct SigmoidNonlin {
  static constexpr const char *kind = "Sigmoid";
  static inline Float nonlin(Float x) { return sigmoid(x); }
  static inline Float yderiv(Float y) { return y * (1 - y); }
};
struct TanhNonlin {
  static constexpr const char *kind = "Tanh";
  static inline Float nonlin(Float x) { return tanh(x); }
  static inline Float yderiv(Float y) { return 1 - y * y; }
};
struct ReluNonlin {
  static constexpr const char *kind = "Relu";
  static inline Float nonlin(Float x) { return relu_(x); }
  static inline Float yderiv(Float y) { return heavi_(y); }
};

void forward_stack(Batch &z, Batch &x, Batch &y);
void backward_stack(Batch &z, Batch &x, Batch &y);
void forward_stack(Batch &z, Batch &x, Sequence &y, int last);
void backward_stack(Batch &z, Batch &x, Sequence &y, int last);

void forward_reverse(Sequence &y, Sequence &x);
void backward_reverse(Sequence &y, Sequence &x);

template <class F>
void forward_full1(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full1(Batch &y, Params &W, Batch &x);

void forward_softmax(Batch &z, Params &W1, Batch &x);
void backward_softmax(Batch &z, Params &W1, Batch &x);
void forward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);
void backward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf);
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf);
template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go);
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go);

void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
template <class F>
void forward_full(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full(Batch &y, Params &W, Batch &x);

void rinit(Batch &m, int no, int ni, Float s, const string mode = "unif", Float offset=0.0);
void rinit(Sequence &m, int no, int ni, Float s, const string mode = "unif", Float offset=0.0);
void rinit(Params &m, int N, int no, int ni, Float s, const string mode = "pos", Float offset=0.0);
bool anynan(Batch &a);
bool anynan(Sequence &a);
bool anynan(Params &a);

template <class S, class T>
inline void transpose(S &dest, T &src) {
  dest.resize(src.dimension(1), src.dimension(0));
  for (int i = 0; i < dest.dimension(0); i++)
    for (int j = 0; j < dest.dimension(1); j++) dest(i, j) = src(j, i);
}

template <class T>
inline void transpose(T &a) {
  T temp;
  transpose(temp, a);
  a = temp;
}
}

#endif
