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
inline Float sum(const EigenTensor1 &m) { return reduction(m.sum()); }
inline Float sum(const EigenTensor2 &m) { return reduction(m.sum()); }

struct Context {
  // The Tensor contains information about which device
  // it is associated with. Only one of these may be set
  // (we could use boost::variant, but we only have
  // two classes, so this is simpler).
  // We still have subtypes for GpuContext and ThreadContext,
  // but because Eigen::Tensor uses ad hoc polymorphism,
  // the main computations need to be expanded inline here.
#ifdef EIGEN_USE_THREADS
  unique_ptr<Eigen::ThreadPoolDevice> tpdev;
#endif
#ifdef EIGEN_USE_GPU
  unique_ptr<Eigen::GpuDevice> gpudev;
#endif

  template <class LHS, class RHS>
  void assign(LHS lhs, RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) lhs.device(*tpdev) = rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) lhs.device(*gpudev) = rhs;
#endif
    else lhs = rhs;
  }

  template <class LHS, class RHS>
  void increment(LHS lhs, RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) lhs.device(*tpdev) += rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) lhs.device(*gpudev) += rhs;
#endif
    else lhs += rhs;
  }

  template <class LHS, class RHS>
  void scale(LHS lhs, RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) lhs.device(*tpdev) *= rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) lhs.device(*gpudev) *= rhs;
#endif
    else lhs *= rhs;
  }

  template <class LHS, class RHS>
  void decrement(LHS lhs, RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) lhs.device(*tpdev) -= rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) lhs.device(*gpudev) -= rhs;
#endif
    else lhs -= rhs;
  }

  template <class LHS, class RHS>
  void invscale(LHS lhs, RHS rhs) {
    if (0) ;
#ifdef EIGEN_USE_THREADS
    else if(tpdev) lhs.device(*tpdev) /= rhs;
#endif
#ifdef EIGEN_USE_GPU
    else if(gpudev) lhs.device(*gpudev) /= rhs;
#endif
    else lhs /= rhs;
  }
  virtual bool isgpu() {
    return false;
  }
  virtual void *malloc(int n) {
    return ::malloc(n);
  }
  virtual void free(void *p) {
    ::free(p);
  }
  virtual void memcpyToDevice(void *dest, void *src, int nbytes) {
    memcpy(dest, src, nbytes);
  }
  virtual void memcpyFromDevice(void *dest, void *src, int nbytes) {
    memcpy(dest, src, nbytes);
  }
  virtual void memcpyDevice(void *dest, void *src, int nbytes) {
    memcpy(dest, src, nbytes);
  }
};


#ifdef EIGEN_USE_THREADS
struct ThreadedContext : public Context {
  unique_ptr<Eigen::ThreadPool> pool;
  ThreadedContext(int n) {
    pool.reset(new Eigen::ThreadPool(n));
    tpdev.reset(new Eigen::ThreadPoolDevice(pool.get(), n));
  }
};
#endif

#ifdef EIGEN_USE_GPU
struct GpuContext : public Context {
  unique_ptr<Eigen::CudaStreamDevice> stream;
  GpuContext() {
    gpudev.reset(new Eigen::GpuDevice(stream.get()));
  }
  bool isgpu() {
    return true;
  }
  void *malloc(int n) {
    void *p = nullptr;
    cudaMalloc(&p, n);
    return p;
  }
  void free(void *p) {
    cudaFree(p);
  }
  void memcpyToDevice(void *dest, void *src, int nbytes) {
    cudaMemcpy(dest, src, nbytes, cudaMemcpyHostToDevice);
  }
  void memcpyFromDevice(void *dest, void *src, int nbytes) {
    cudaMemcpy(dest, src, nbytes, cudaMemcpyDeviceToHost);
  }
  void memcpyDevice(void *dest, void *src, int nbytes) {
    cudaMemcpy(dest, src, nbytes, cudaMemcpyDeviceToDevice);
  }
};
#endif

extern Context *default_context;

// A simple Tensor class that handles multiple device
// types a bit more transparently. It handles allocation/deallocation,
// plus assignment.

struct Tensor2 {
  Context *context = default_context;

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
    context->free(ptr);
    ptr = nullptr;
    dims[0] = 0;
    dims[1] = 0;
  }
  void resize(int n, int m) {
    clear();
    ptr = (Float*)context->malloc(n * m * sizeof(Float));
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

  template <class RHS>
  void operator=(RHS rhs) {
    context->assign(map(), rhs);
  }
  template <class RHS>
  void operator+=(RHS rhs) {
    context->increment(map(), rhs);
  }

  Float &operator()(int i, int j) {
    return (**this)(i,j);
  }
  void operator=(TensorMap2 other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    context->memcpyToDevice(ptr, other.data(), nbytes);
  }
  void operator=(const Tensor2 &other) {
    resize(other.dimension(0), other.dimension(1));
    int nbytes = total_size() * sizeof(Float);
    if (context->isgpu()) {
      if(other.context->isgpu()) {
        context->memcpyDevice(ptr, other.ptr, nbytes);
      } else {
        context->memcpyToDevice(ptr, other.ptr, nbytes);
      }
    } else {
      if(other.context->isgpu()) {
        context->memcpyFromDevice(ptr, other.ptr, nbytes);
      } else {
        memcpy(ptr, other.ptr, nbytes);
      }
    }
  }
  void setConstant(int n, int m, Float c) {
    resize(n,m);
    map().setConstant(c);
  }
  void setZero(int n, int m) {
    setConstant(n, m, 0);
  }
  void setZero() {
    map().setZero();
  }
};

// This is a bit of syntactic sugar that allows us to handle
// devices for complex LHS expression. For example,
//
//    mytensor>> mytensor.slice(...) = ... ;
//
// Under the covers, this will generate code similar to this:
//
//    if(gpudev) mytensor.slice(...).device(*gpudev) = ... ;
//    else if(threaddev) mytensor.slice(...).device(*threaddev) = ... ;
//    else mytensor.slice(...) = ... ;
//
// Ordinarily, in C++, we would use inheritance and virtual functions,
// but that doesn't mesh with Eigen::Tensor's use of ad-hoc polymorphism
// for device-specific code generation.

template <class LHS>
struct ContextSetter {
  Context *context;
  LHS lhs;
  ContextSetter(Context *context, LHS lhs) : context(context), lhs(lhs) {}
  template <class RHS>
  void operator=(RHS rhs) {
    context->assign(lhs, rhs);
  };
  template <class RHS>
  void operator+=(RHS rhs) {
    context->increment(lhs, rhs);
  };
};

template <class LHS>
ContextSetter<LHS> operator>>(Context &context, LHS lhs) {
  return ContextSetter<LHS>(&context, lhs);
}

template <class LHS>
ContextSetter<LHS> operator>>(Context *context, LHS lhs) {
  return ContextSetter<LHS>(context, lhs);
}

template <class LHS>
ContextSetter<LHS> operator>>(Tensor2 &tensor, LHS lhs) {
  return ContextSetter<LHS>(tensor.context, lhs);
}

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

// Nonlinearities (for parameterizing layers).

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


}

#endif
