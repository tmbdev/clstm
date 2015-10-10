#ifndef clstm_compute__
#define clstm_compute__

#include <vector>
#include <Eigen/Dense>

namespace ocropus {
using namespace std;

#define ROWS(A) (A).rows()
#define COLS(A) (A).cols()
#define MAPFUN(M, F) ((M).unaryExpr(ptr_fun(F)))

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
  void operator=(T other) { (Mat &)*this = other; }
  void zeroGrad() { d.setZero(rows(), cols()); }
};
struct Params : Mat {
  Mat d;
  template <class T>
  void operator=(T other) { (Mat &)*this = other; }
  void zeroGrad() { d.setZero(rows(), cols()); }
  void update(Float lr, Float mom) {
    *this += lr*d;
    d *= mom;
  }
};

// typedef vector<Mat> Sequence;
struct Sequence {
  vector<Batch> steps;
  Sequence() {}
  Sequence(int n) : steps(n) {}
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
  int size() const {
    return steps.size();
  }
  void resize(int n) {
    resize(n, 1, 1);
  }
  void resize(int n, int rows, int cols) {
    steps.resize(n);
    for (int t = 0; t < n; t++) {
      steps[t].setZero(rows, cols);
      steps[t].d.setZero(rows, cols);
    }
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

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

void gradient_clip(Sequence &s, Float m = 100.0);
void gradient_clip(Batch &b, Float m = 100.0);
void gradient_clip(Mat &d, Float m = 100.0);

// FIXME: refactor into forward_/backward_
struct NoNonlin {
  static constexpr const char *kind = "Linear";
  template <class T>
  static void f(T &x) {}
  template <class T, class U>
  static void df(T &dx, U &y) {}
};

struct SigmoidNonlin {
  static constexpr const char *kind = "Sigmoid";
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
void forward_stack(Batch &z, Batch &x, Sequence &y, int last);
void backward_stack(Batch &z, Batch &x, Sequence &y, int last);

void forward_reverse(Sequence &y, Sequence &x);
void backward_reverse(Sequence &y, Sequence &x);

template <class F>
void forward_full1(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full1(Batch &y, Params &W, Batch &x, Float gc);

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

// FIXME: replace these in LSTM; eliminate gradient_clip here
void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
template <class F>
void forward_full(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc);

void randgauss(Mat &m);
void randgauss(Vec &v);
void randinit(Mat &m, float s, const string mode = "unif");
void randinit(Vec &m, float s, const string mode = "unif");
void randinit(Mat &m, int no, int ni, float s, const string mode = "unif");
void randinit(Vec &m, int no, float s, const string mode = "unif");
void zeroinit(Mat &m, int no, int ni);
void zeroinit(Vec &m, int no);
void resize(Sequence &seq, int nsteps, int dims, int bs);
int size(Sequence &seq, int dim);
Vec timeslice(const Sequence &s, int i, int b = 0);

bool anynan(Batch &a);
bool anynan(Sequence &a);
}

#endif
