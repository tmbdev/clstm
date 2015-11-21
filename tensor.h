#ifndef ocropus_tensor_
#define ocropus_tensor_

#include <memory>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#ifdef CLSTM_CUDA
#include "cuda_runtime.h"
#include "cuda.h"
#endif

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

inline Float sigmoid(Float x) { return 1.0 / (1.0 + limexp(-x)); }

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

typedef Eigen::Matrix<Float, Eigen::Dynamic, 1> EigenVector;
typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
typedef Eigen::Map<EigenVector> VectorMap;
typedef Eigen::Map<EigenMatrix> MatrixMap;

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

// A simple Tensor class that handles multiple device
// types a bit more transparently. It handles allocation/deallocation,
// plus assignment.

struct Tensor2 {
 protected:
  int gpu = -1;

 public:
  // The data and dimensions of this tensor.

  int dims[2] = {0, 0};
  Float *ptr = nullptr;

  // The tensor data may be owned (usually by a Sequence);
  // in that case, it isn't deallocated when the tensor goes
  // out of scope.
  void *owner = nullptr;
  bool resizeable = true;

  Tensor2() {}
  Tensor2(const Tensor2 &other) { *this = other; }
  ~Tensor2() { reset(); }
  void reset() {
    if (!ptr) return;
    if (!owner) {
      if (gpu < 0) {
	free(ptr);
      } else {
#ifdef CLSTM_CUDA
	cudaFree(ptr);
#endif
      }
    }
    ptr = nullptr;
    dims[0] = 0;
    dims[1] = 0;
  }
  int getGpu() { return gpu; }
  void setGpu(int n) {
    reset();
#ifdef CLSTM_CUDA
    gpu = n;
#else
    assert(n < 0);
#endif
  }
  void resize(int n, int m) {
    assert(ptr == nullptr || (dims[0] > 0 && dims[1] > 0));
    // resizing to the same size is always allowed
    if (dims[0] == n && dims[1] == m) return;
    assert(resizeable);
    assert(ptr==nullptr || !owner);
    reset();
    if (n == 0 || m == 0) return;
    dims[0] = n;
    dims[1] = m;
    if (gpu < 0) {
      ptr = (Float *)malloc(n * m * sizeof(Float));
    } else {
#ifdef CLSTM_CUDA
      void *p;
      cudaMalloc(&p, n * m * sizeof(Float));
      ptr = (Float *)p;
#else
      assert(false && "not compiled for CUDA");
#endif
    }
  }
  void like(Tensor2 &other) {
    setGpu(other.getGpu());
    resize(other.dimension(0), other.dimension(1));
  }
  void like(TensorMap2 other) {
    resize(other.dimension(0), other.dimension(1));
  }
  Float *data() { return ptr; }

  int dimension(int i) const { return dims[i]; }
  int rows() { return dims[0]; }
  int cols() { return dims[1]; }
  int total_size() { return dims[0] * dims[1]; }

  // These operators allow easy access to the TensorMap version
  // of the tensor. Use these on the right hand side of an expression.
  // The following are equivalent:
  //
  //   *x, x(), x.map()
  //
  // Probably the x() is the most natural one to use in most expressions.
  TensorMap2 operator*() { return TensorMap2(ptr, dims[0], dims[1]); }
  TensorMap2 operator()() { return **this; }
  TensorMap2 map() { return **this; }

  // Convert the tensor to an Eigen Matrix Map
  MatrixMap mat() { return MatrixMap(ptr, dims[0], dims[1]); }

  // Extract the offset and matrix part from a homogeneous
  // matrix transformation. This can also be expressed
  // using slice/chip in Eigen::Tensor, but that turns
  // out to be significantly slower.
  TensorMap2 map1() { return TensorMap2(ptr + dims[0], dims[0], dims[1] - 1); }
  TensorMap1 off1() { return TensorMap1(ptr, dims[0]); }

  // Extract the offset and matrix part from a homogeneous
  // matrix transformation, this time using Eigen::Matrix
  // types.
  MatrixMap mat1() { return MatrixMap(ptr + dims[0], dims[0], dims[1] - 1); }
  VectorMap vec1() { return VectorMap(ptr, dims[0]); }

  Float &operator()(int i, int j) {
    assert(gpu < 0 && "use get() for gpu access");
    return (**this)(i, j);
  }

  // accessors that work on GPU
  Float get(int i, int j) {
    if (gpu < 0) {
      return (**this)(i, j);
    } else {
#ifdef CLSTM_CUDA
      Float *devptr = ptr + (i + j * dims[0]);
      Float value;
      cudaMemcpy(&value, devptr, sizeof(Float), cudaMemcpyDeviceToHost);
      return value;
#else
      THROW("not compiled for GPU");
#endif
    }
  }
  void put(Float value, int i, int j) {
    if (gpu < 0) {
      (**this)(i, j) = value;
    } else {
#ifdef CLSTM_CUDA
      Float *devptr = ptr + (i + j * dims[0]);
      cudaMemcpy(devptr, &value, sizeof(Float), cudaMemcpyDeviceToHost);
#else
      THROW("not compiled for GPU");
#endif
    }
  }
  void operator=(TensorMap2 other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    memcpy(ptr, other.data(), nbytes);
  }
  void operator=(const Tensor2 &other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
#ifdef CLSTM_CUDA
    if (gpu >= 0 && other.gpu >= 0) {
      cudaMemcpy(ptr, other.ptr, nbytes, cudaMemcpyDeviceToDevice);
    } else if (gpu >= 0 && other.gpu < 0) {
      cudaMemcpy(ptr, other.ptr, nbytes, cudaMemcpyHostToDevice);
    } else if (gpu < 0 && other.gpu >= 0) {
      cudaMemcpy(ptr, other.ptr, nbytes, cudaMemcpyDeviceToHost);
#else
    if (0) {
#endif
    } else {
      memcpy(ptr, other.ptr, nbytes);
    }
  }
  void setZero() {
#ifdef CLSTM_CUDA
    if (gpu >= 0) {
      cudaMemset(ptr, 0, total_size() * sizeof(Float));
#else
    if (0) {
#endif
    } else {
      int N = total_size();
      for (int i = 0; i < N; i++) ptr[i] = 0;
    }
  }
  void setZero(int n, int m) {
    resize(n, m);
    setZero();
  }
};

inline Float asum1(const TensorRef1 &a) {
  int n = a.dimension(0);
  Float result = 0.0;
  for (int i = 0; i < n; i++) result += a(i);
  return result;
}
inline Float asum2(const TensorRef2 &a) {
  int n = a.dimension(0);
  int m = a.dimension(1);
  Float result = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) result += a(i, j);
  return result;
}
inline Float amax1(const TensorRef1 &a) {
  int n = a.dimension(0);
  Float result = a(0, 0);
  for (int i = 0; i < n; i++) result = fmax(result, a(i));
  return result;
}
inline Float amax2(const TensorRef2 &a) {
  int n = a.dimension(0);
  int m = a.dimension(1);
  Float result = a(0, 0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) result = fmax(result, a(i, j));
  return result;
}
}

#endif
