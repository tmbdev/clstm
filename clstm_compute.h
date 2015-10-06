#ifndef clstm_compute__
#define clstm_compute__

#include <vector>
#include <Eigen/Dense>

namespace ocropus {
using namespace std;

#ifdef LSTM_DOUBLE
typedef double Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
#else
typedef float Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXf Vec;
typedef Eigen::MatrixXf Mat;
#endif

template <typename F, typename T>
void each(F f, T &a) {
  f(a);
}
template <typename F, typename T, typename... Args>
void each(F f, T &a, Args &&... args) {
  f(a);
  each(f, args...);
}

#ifndef MAXEXP
#define MAXEXP 30
#endif

inline Float tanh_(Float x) { return tanh(x); }
inline Float relu_(Float x) { return x <= 0 ? 0 : x; }
inline Float heavi_(Float x) { return x <= 0 ? 0 : 1; }

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

// These macros define the major matrix operations used
// in CLSTM. They are here for eventually converting the
// inner loops of CLSTM from Eigen::Matrix to Eigen::Tensor
// (which uses different and incompatible notation)
//
// NB: In C++ 14, we can write Eigen functions more easily like this:
// auto HOMDOT(Mat &A1, Mat &B) {return (DOT(CBUTFIRST(A1), B).colwise() +
// CFIRST(A1));}
//
// All of this will be cleaned up when we're switching to Eigen::Tensor

#define DOT(M, V) ((M) * (V))
#define MATMUL(A, B) ((A) * (B))
#define MATMUL_TR(A, B) ((A).transpose() * (B))
#define MATMUL_RT(A, B) ((A) * (B).transpose())
#define EMUL(U, V) ((U).array() * (V).array()).matrix()
#define EMULV(U, V) ((U).array() * (V).array()).matrix()
#define TRANPOSE(U) ((U).transpose())
#define ROWS(A) (A).rows()
#define COLS(A) (A).cols()
#define COL(A, b) (A).col(b)
#define MAPFUN(M, F) ((M).unaryExpr(ptr_fun(F)))
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

template <class NONLIN, class T>
inline Mat nonlin(T &a) {
  Mat result = a;
  NONLIN::f(result);
  return result;
}
template <class NONLIN, class T>
inline Mat yprime(T &a) {
  Mat result = Mat::Ones(ROWS(a), COLS(a));
  NONLIN::df(result, a);
  return result;
}
template <class NONLIN, class T>
inline Mat xprime(T &a) {
  Mat result = Mat::Ones(ROWS(a), COLS(a));
  Mat temp = a;
  NONLIN::f(temp);
  NONLIN::df(result, temp);
  return result;
}

struct Batch : Mat {
  Mat d;
  template <class T>
  void operator=(T other) {
    (Mat &)*this = other;
    // d.setZero(2,3);  // invalidate it
  }
  void zeroGrad() { d.setZero(rows(), cols()); }
};
typedef Batch Params;

// typedef vector<Mat> Sequence;
struct Sequence {
  vector<Batch> steps;
  Sequence() {}
  Sequence(int n) : steps(n) {}
  void clear() { steps.clear(); }
  int size() const { return steps.size(); }
  void resize(int n) { steps.resize(n); }
  int rows() { return steps[0].rows(); }
  int cols() { return steps[0].cols(); }
  void resize(int n, int rows, int cols) {
    steps.resize(n);
    for (int t = 0; t < n; t++) steps[t].resize(rows, cols);
  }
  void copy(const Sequence &other) {
    resize(other.size());
    for (int t = 0; t < other.size(); t++) steps[t] = other[t];
  }
  Batch &operator[](int i) { return steps[i]; }
  const Batch &operator[](int i) const { return steps[i]; }
  void zero() {
    for (int t = 0; t < steps.size(); t++) steps[t].setZero();
  }
  void zeroGrad() {
    for (int t = 0; t < steps.size(); t++) steps[t].zeroGrad();
  }
};

void gradient_clip(Sequence &s, Float m = 1.0);
void gradient_clip(Mat &d, Float m = 1.0);

struct NoNonlin {
  static constexpr const char *kind = "Linear";
  static constexpr const char *name = "linear";
  template <class T>
  static void f(T &x) {}
  template <class T, class U>
  static void df(T &dx, U &y) {}
};

struct SigmoidNonlin {
  static constexpr const char *kind = "Sigmoid";
  static constexpr const char *name = "sigmoid";
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, sigmoid);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= y.array() * (1 - y.array());
  }
};
struct TanhNonlin {
  static constexpr const char *kind = "Tanh";
  static constexpr const char *name = "tanh";
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, tanh_);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= (1 - y.array().square());
  }
};
struct ReluNonlin {
  static constexpr const char *kind = "Relu";
  static constexpr const char *name = "relu";
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, relu_);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= MAPFUN(y, heavi_).array();
  }
};

void forward_stack(Batch &z, Batch &x, Batch &y);
void backward_stack(Batch &z, Batch &x, Batch &y);

void forward_reverse(Sequence &y, Sequence &x);
void backward_reverse(Sequence &y, Sequence &x);

void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last);

template <class F>
void forward_full1(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full1(Batch &y, Params &W, Batch &x, Float gc);

template <class F>
void forward_full(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc);

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf);
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf);
template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go);
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go);

}

#endif
