#ifndef ocropus_tensor_
#define ocropus_tensor_

#include <memory>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ocropus {

using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::TensorRef;
using Eigen::DSizes;
using Eigen::Index;
using Eigen::array;
using std::shared_ptr;
using std::unique_ptr;

#define ROWS(A) (A).rows()
#define COLS(A) (A).cols()
//#define MAPFUN(M, F) ((M).unaryExpr(ptr_fun(F)))

#ifdef LSTM_DOUBLE
typedef double Float;
#else
typedef float Float;
#endif

// Mathematical helper functions.

inline Float tanh_(Float x) { return tanh(x); }
inline Float relu_(Float x) { return x <= 0 ? 0 : x; }
inline Float heavi_(Float x) { return x <= 0 ? 0 : 1; }

#ifndef MAXEXP
#define MAXEXP 30
#endif

inline Float limexp(Float x) {
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
}

inline Float sigmoid(Float x) {
  return 1.0 / (1.0 + limexp(-x));
}

inline Float log_add(Float x, Float y) {
  if (abs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline Float log_mul(Float x, Float y) { return x + y; }

using Eigen::Tensor;
using Eigen::TensorMap;

typedef Float Scalar;
typedef Eigen::Tensor<Float, 1> EigenTensor1;
typedef Eigen::Tensor<Float, 2> EigenTensor2;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 1>> TensorMap1;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 2>> TensorMap2;
typedef Eigen::TensorRef<Eigen::Tensor<Float, 1>> TensorRef1;
typedef Eigen::TensorRef<Eigen::Tensor<Float, 2>> TensorRef2;

inline int rows(const TensorMap2 &m) { return m.dimension(0); }
inline int cols(const TensorMap2 &m) { return m.dimension(1); }
inline int rows(const EigenTensor2 &m) { return m.dimension(0); }
inline int cols(const EigenTensor2 &m) { return m.dimension(1); }

// inline Float reduction(const EigenTensor1 &m) { return m(0); }
inline Float reduction(const EigenTensor1 &m) { return m(0); }
inline Float reduction(const TensorMap1 &m) { return m(0); }
inline Float reduction(Float m) { return m; }
inline int argmax(const EigenTensor1 &m) {
  int mi = -1;
  Float mv = m(0);
  for (int i = 0; i < m.dimension(0); i++) {
    if (m(i) < mv) continue;
    mi = i;
    mv = m(i);
  }
  return mi;
}
//inline Float sum(const EigenTensor1 &m) { return reduction(m.sum()); }
//inline Float sum(const EigenTensor2 &m) { return reduction(m.sum()); }

// A simple Tensor class that handles multiple device
// types a bit more transparently. It handles allocation/deallocation,
// plus assignment.

struct Tensor2 {
  // The data and dimensions of this tensor. Data is always
  // heap allocated and not shared.
  int dims[2];
  Float *ptr = nullptr;

  Tensor2() {}
  Tensor2(const Tensor2 &other) {
    *this = other;
  }
  Tensor2(TensorRef<Tensor<Float,2>> other) { }
  ~Tensor2() {
    clear();
  }
  void clear() {
    if(!ptr) return;
    free(ptr);
    ptr = nullptr;
    dims[0] = 0;
    dims[1] = 0;
  }
  void resize(int n, int m) {
    clear();
    ptr = (Float*)malloc(n * m * sizeof(Float));
    dims[0] = n;
    dims[1] = m;
  }
  void like(TensorMap2 other) {
    resize(other.dimension(0), other.dimension(1));
  }
  Float *data() {
    return ptr;
  }

  int dimension(int i) const {
    return dims[i];
  }
  int rows() {
    return dims[0];
  }
  int cols() {
    return dims[1];
  }
  int total_size() {
    return dims[0] * dims[1];
  }

  // These operators allow easy access to the TensorMap version
  // of the tensor. Use these on the right hand side of an expression.
  // The following are equivalent:
  //
  //   *x, x(), x.map()
  //
  // Probably the x() is the most natural one to use in most expressions.

  TensorMap2 operator*() {
    return TensorMap2(ptr, dims[0], dims[1]);
  }
  TensorMap2 operator()() {
    return **this;
  }
  TensorMap2 map() {
    return **this;
  }

  Float &operator()(int i, int j) {
    return (**this)(i,j);
  }
  void operator=(TensorMap2 other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    memcpy(ptr, other.data(), nbytes);
  }
  void operator=(const Tensor2 &other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    memcpy(ptr, other.ptr, nbytes);
  }
  void setConstant(Float c) {
    int N = total_size();
    for(int i=0; i<N; i++) ptr[i] = c;
  }
  void setConstant(int n, int m, Float c) {
    resize(n,m);
    setConstant(c);
  }
  void setZero(int n, int m) {
    resize(n,m);
    setConstant(0);
  }
  void setZero() {
    setConstant(0);
  }
};

}

#endif
