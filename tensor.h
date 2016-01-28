#ifndef ocropus_tensor_
#define ocropus_tensor_

// A tensor storage class for computations within CLSTM.
// There are several tensor and matrix classes used within CLSTM:
//
// - Tensor2: the basic CLSTM tensor class
// - TensorMap2: the Eigen TensorMap class (used for Eigen computations)
// - EigenTensor2: native Eigen storage, only works on CPU
// - MatrixMap: Eigen mapping of Matrix used for CPU computations (slightly
// faster)
//
// Other storage:
//
// - Batch: a pair of Tensor2 instances for activations and derivatives
// - Params: a pair of Tensor2 instances for weights and derivatives
// - Sequence: a sequence of Batches, may use a rank 3 tensor internally
//
// Why is this so complicated? There are a number of constraints:
//
// - We need to use Eigen::Tensor for compatibility with TensorFlow
// - Eigen::Tensor uses overloading and templates for genericity
// - We want to compile only GPU code with nvcc (since nvcc is otherwise buggy)
// - We don't want two separate versions of the numerical functions.
// - The code was developed incrementally from a non-GPU version.
// - The GPU code imposes some special constraints in order to run efficiently.
// - Eigen::Tensor is slow for some cases where Eigen::Matrix is fast.
//
// You can think of Tensor2 as a rank-2 tensor with shape (w,h), Batch
// as a rank-3 tensor with shape (w,h,2), and Sequence as a rank-4
// tensor with shape (w,h,2,N). In standard tensor libraries, we would
// simply express the various computations as computations over
// slices, bit the Eigen Tensor library does not have full support for
// tensor slices as independent objects.

#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef CLSTM_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#endif

namespace ocropus {

using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::TensorRef;
using Eigen::DSizes;
// using Eigen::Index;
typedef ptrdiff_t Index;
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
  if (fabs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline Float log_mul(Float x, Float y) { return x + y; }

using Eigen::Tensor;
using Eigen::TensorMap;

typedef Float Scalar;
typedef Eigen::Tensor<Float, 1> EigenTensor1;
typedef Eigen::Tensor<Float, 2> EigenTensor2;
typedef Eigen::Tensor<Float, 3> EigenTensor3;
typedef Eigen::Tensor<Float, 4> EigenTensor4;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 1>> TensorMap1;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 2>> TensorMap2;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 3>> TensorMap3;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 4>> TensorMap4;
typedef Eigen::TensorRef<Eigen::Tensor<Float, 1>> TensorRef1;
typedef Eigen::TensorRef<Eigen::Tensor<Float, 2>> TensorRef2;
typedef Eigen::TensorRef<Eigen::Tensor<Float, 3>> TensorRef3;
typedef Eigen::TensorRef<Eigen::Tensor<Float, 4>> TensorRef4;

typedef Eigen::Matrix<Float, Eigen::Dynamic, 1> EigenVector;
typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
typedef Eigen::Map<EigenVector> VectorMap;
typedef Eigen::Map<EigenMatrix> MatrixMap;

#if 0
inline int rows(const TensorMap2 &m) { return m.dimension(0); }
inline int cols(const TensorMap2 &m) { return m.dimension(1); }
inline int rows(const EigenTensor2 &m) { return m.dimension(0); }
inline int cols(const EigenTensor2 &m) { return m.dimension(1); }
#endif

template <class T>
inline void alloc_gpu(T *&p, int nbytes, int gpu) {
  p = nullptr;
#ifdef CLSTM_CUDA
  if (gpu < 0) {
    p = (T *)malloc(nbytes);
  } else {
    cudaMalloc((void **)&p, nbytes);
  }
#else
  assert(gpu < 0 && "not compiled for CUDA");
  p = (T *)malloc(nbytes);
#endif
}

template <class T>
inline void free_gpu(T *&p, int gpu) {
#ifdef CLSTM_CUDA
  if (gpu < 0) {
    free((void *)p);
    p = nullptr;
  } else {
    cudaFree((void **)&p);
    p = nullptr;
  }
#else
  assert(gpu < 0 && "not compiled for CUDA");
  free((void *)p);
  p = nullptr;
#endif
}

inline void memcpy_gpu(void *dest, int dest_gpu, void *src, int src_gpu,
                       int nbytes) {
#ifdef CLSTM_CUDA
  if (dest_gpu >= 0 && src_gpu >= 0) {
    cudaMemcpy(dest, src, nbytes, cudaMemcpyDeviceToDevice);
  } else if (dest_gpu >= 0 && src_gpu < 0) {
    cudaMemcpy(dest, src, nbytes, cudaMemcpyHostToDevice);
  } else if (dest_gpu < 0 && src_gpu >= 0) {
    cudaMemcpy(dest, src, nbytes, cudaMemcpyDeviceToHost);
  } else {
    memcpy(dest, src, nbytes);
  }
#else
  assert(dest_gpu < 0 && src_gpu < 0);
  memcpy(dest, src, nbytes);
#endif
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
  bool displaced = false;
  bool resizeable = true;

  Tensor2() {}
  Tensor2(const Tensor2 &other) { *this = other; }
  ~Tensor2() { reset(); }
  void displaceTo(Float *ptr, int n, int m, int gpu = -1) {
    displaced = true;
    this->gpu = gpu;
    this->ptr = ptr;
    dims[0] = n;
    dims[1] = m;
  }
  void reset() {
    if (!ptr) return;
    if (!displaced) free_gpu(ptr, gpu);
    displaced = false;
    ptr = nullptr;
    dims[0] = 0;
    dims[1] = 0;
  }
  int getGpu() const { return gpu; }
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
    assert(ptr == nullptr || !displaced);
    reset();
    if (n == 0 || m == 0) return;
    dims[0] = n;
    dims[1] = m;
    alloc_gpu(ptr, n * m * sizeof(Float), gpu);
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
  int rows() const { return dims[0]; }
  int cols() const { return dims[1]; }
  int total_size() const { return dims[0] * dims[1]; }

  // These operators allow easy access to the TensorMap version
  // of the tensor. Use these on the right hand side of an expression.
  // The following are equivalent:
  //
  //   *x, x(), x.map()
  //
  // Probably the x() is the most natural one to use in most expressions.
  TensorMap2 operator*() const { return TensorMap2(ptr, dims[0], dims[1]); }
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

  const Float &operator()(int i, int j) const {
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
      return 0;
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
  int nbytes() const { return total_size() * sizeof(Float); }
  void operator=(TensorMap2 other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    memcpy(ptr, other.data(), nbytes);
  }
  void operator=(const Tensor2 &other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    memcpy_gpu(ptr, gpu, other.ptr, other.gpu, nbytes);
  }
  void setZero() {
#ifdef CLSTM_CUDA
    if (gpu >= 0)
      cudaMemset(ptr, 0, total_size() * sizeof(Float));
    else
      memset(ptr, 0, total_size() * sizeof(Float));
#else
    memset(ptr, 0, total_size() * sizeof(Float));
#endif
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
inline int argmax(const TensorRef1 &m) {
  int mi = -1;
  Float mv = m(0);
  for (int i = 0; i < m.dimension(0); i++) {
    if (m(i) < mv) continue;
    mi = i;
    mv = m(i);
  }
  return mi;
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
