#ifndef ocropus_tensor_
#define ocropus_tensor_

#include <memory>
#include <unordered_map>
#include <Eigen/Dense>
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
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
#else
typedef float Float;
typedef Eigen::MatrixXf Mat;
typedef Eigen::VectorXf Vec;
#endif

typedef Eigen::Map<Mat> MatMap;

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
inline Float sum(const EigenTensor1 &m) { return reduction(m.sum()); }
inline Float sum(const EigenTensor2 &m) { return reduction(m.sum()); }

// A simple Tensor class that handles multiple device
// types a bit more transparently. It handles allocation/deallocation,
// plus assignment.

struct Tensor2 {
  Eigen::ThreadPoolDevice *tpdev = nullptr;
  Eigen::GpuDevice *gpudev = nullptr;
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

  TensorMap2 operator*() {
    return TensorMap2(ptr, dims[0], dims[1]);
  }
  TensorMap2 operator()() {
    return **this;
  }
  TensorMap2 map() {
    return **this;
  }

  // For convenience, we're forwarding a bunch of assignment-related
  // operators so that we can wrap them up with the .device(...)
  // call.
  template <class RHS>
  void operator=(RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) map().device(*tpdev) = rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) map().device(*gpudev) = rhs;
#endif
    else map() = rhs;
  }
  template <class RHS>
  void operator+=(RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) map().device(*tpdev) += rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) map().device(*gpudev) += rhs;
#endif
    else map() += rhs;
  }

  Float &operator()(int i, int j) {
    return (**this)(i,j);
  }
  void operator=(const Tensor2 &other) {
    resize(other.dimension(0), other.dimension(1));
    memcpy(ptr, other.ptr, total_size() * sizeof(Float));
  }
  void setConstant(int n, int m, Float c) {
    resize(n,m);
    for(int N=n*m, i=0; i<N; i++) ptr[i] = c;
  }
  void setZero(int n, int m) {
    setConstant(n, m, 0);
  }
  void setZero() {
    for(int N=rows()*cols(), i=0; i<N; i++) ptr[i] = 0;
  }
};

// This is a bit of syntactic sugar that allows us to handle
// devices for complex LHS expression. For example,
//
//    mytensor>> mytensor.slice(...) = ... ;
//
// This is a bit roundabout because of the way Eigen handles
// devices.

template <class LHS>
struct ContextSetter {
  Tensor2 *context;
  LHS lhs;
  ContextSetter(Tensor2 *context, LHS lhs) : context(context), lhs(lhs) {}
  template <class RHS>
  void operator=(RHS rhs) {
    if (0) {
#ifdef EIGEN_USE_THREADS
    } else if (context->tpdev) {
      lhs.device(*context->tpdev) = rhs;
#endif
#ifdef EIGEN_USE_GPU
    } else if (context->gpudev) {
      lhs.device(*context->gpudev) = rhs;
#endif
    } else {
      lhs = rhs;
    }
  };
};

template <class LHS>
ContextSetter<LHS> operator>>(Tensor2 &context, LHS lhs) {
  return ContextSetter<LHS>(&context, lhs);
}

}

#endif
