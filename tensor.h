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
typedef Eigen::Tensor<Float, 1> Tensor1;
typedef Eigen::Tensor<Float, 2> Tensor2;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 1>> Ten1;
typedef Eigen::TensorMap<Eigen::Tensor<Float, 2>> Ten2;

inline int rows(const Ten2 &m) { return m.dimension(0); }
inline int cols(const Ten2 &m) { return m.dimension(1); }
inline int size(const Ten1 &m) { return m.dimension(0); }
inline int rows(const Ten1 &m) { return m.dimension(0); }
inline int cols(const Ten1 &m) { THROW("cols applied to Ten1"); }
inline int rows(const Tensor2 &m) { return m.dimension(0); }
inline int cols(const Tensor2 &m) { return m.dimension(1); }
inline int size(const Tensor1 &m) { return m.dimension(0); }
inline int rows(const Tensor1 &m) { return m.dimension(0); }
inline int cols(const Tensor1 &m) { THROW("cols applied to Ten1"); }

inline Float reduction_(const Tensor1 &m) { return m(0); }
inline Float reduction_(const Ten1 &m) { return m(0); }
inline Float reduction_(float m) { return m; }
inline Float maximum(const Tensor1 &m) { return reduction_(m.maximum()); }
inline Float maximum(const Tensor2 &m) { return reduction_(m.maximum()); }
inline int argmax(const Tensor1 &m) {
  int mi = -1;
  Float mv = m(0);
  for (int i = 0; i < size(m); i++) {
    if (m(i) < mv) continue;
    mi = i;
    mv = m(i);
  }
  return mi;
}
inline Float sum(const Tensor1 &m) { return reduction_(m.sum()); }
inline Float sum(const Tensor2 &m) { return reduction_(m.sum()); }
typedef Eigen::Map<Eigen::MatrixXf> MatrixMap;

// helper functions for Eigen::Tensor axes and sizes

inline Eigen::array<Eigen::IndexPair<int>, 1> axes(int i, int j) {
  Eigen::array<Eigen::IndexPair<int>, 1> result = {Eigen::IndexPair<int>(i, j)};
  return result;
}

inline Eigen::array<ptrdiff_t, 1> ar(int i) { return Eigen::array<ptrdiff_t, 1>({i}); }

inline Eigen::array<ptrdiff_t, 2> ar(int i, int j) {
  return Eigen::array<ptrdiff_t, 2>({i, j});
}

inline Eigen::Sizes<1> S(int i) { return Eigen::Sizes<1>({i}); }

inline Eigen::Sizes<2> S(int i, int j) { return Eigen::Sizes<2>({i, j}); }

struct Context {
};
struct ThreadPoolContext : Context {
  Eigen::ThreadPoolDevice *device = nullptr;
};
struct GpuContext : Context {
  Eigen::GpuDevice *device = nullptr;
};

extern std::unordered_map<Float*,int> refcounts;

template <class LHS>
struct ContextSetter {
  Context *context;
  LHS lhs;
  ContextSetter(Context *context, LHS lhs) : context(context), lhs(lhs) {}
  template <class RHS>
  void operator=(RHS rhs) {
    if(!context) {
      lhs = rhs;
    } else if(typeid(context)==typeid(ThreadPoolContext)) {
      Eigen::ThreadPoolDevice *device = dynamic_cast<ThreadPoolContext*>(context)->device;
      lhs.device(*device) = rhs;
    } else if(typeid(context)==typeid(GpuContext)) {
      Eigen::GpuDevice *device = dynamic_cast<GpuContext*>(context)->device;
      lhs.device(*device) = rhs;
    } else {
      THROW("unknown context");
    }
  };
};

template <class LHS>
ContextSetter<LHS> operator>>(Context *context, LHS lhs) {
  return ContextSetter<LHS>(context, lhs);
}

struct tensor2 {
  Eigen::array<int,2> dims;
  Context *context = nullptr;
  Float *ptr = nullptr;

  tensor2() {}
  tensor2(const tensor2 &other) { 
    *this = other;
  }
  tensor2(TensorRef<Tensor<Float,2>> other) { }
  ~tensor2() { 
    decref();
  }

  void incref() {
    if(!ptr) return;
    refcounts[ptr]++;
  }
  void decref() {
    if(!ptr) return;
    if(--refcounts[ptr]>0) return;
    refcounts.erase(ptr);
    free(ptr);
    ptr = nullptr;
  }

  void setContext(Context *context) { 
    decref();
    ptr = nullptr;
    this->context = context;
  }
  void convertToContext(Context *ncontext) {
    THROW("unimplemented");
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
  Float &operator()(int i, int j) {
    return (**this)(i,j);
  }
  void resize(int n, int m) { 
    decref();
    ptr = (Float*)malloc(n * m * sizeof(Float));
    refcounts[ptr] = 1;
    dims[0] = n; 
    dims[1] = m;
  }
  int total_size() {
    return dims[0] * dims[1];
  }
  TensorMap<Tensor<Float,2>> operator*() {
    return TensorMap<Tensor<Float,2>>(ptr, dims[0], dims[1]);
  }
  void operator=(const tensor2 &other) {
    if(other.ptr==nullptr) {
      decref();
      return;
    }
    if(context==nullptr) {
      resize(other.dimension(0), other.dimension(1));
      memcpy(ptr, other.ptr, total_size() * sizeof(Float));
    } else {
      throw "unimplemented";
    }
  }
  template <class RHS>
  void operator=(RHS rhs) {
    **this = rhs;
  }
  void share(tensor2 &other) {
    if(context==other.context) {
      other.incref();
      decref();
      ptr = other.ptr;
    } else {
      throw "unimplemented";
    }
  }
  void take(tensor2 &other) {
    if(context==other.context) {
      other.incref();
      decref();
      ptr = other.ptr;
      other.decref();
      other.ptr = nullptr;
    } else {
      throw "unimplemented";
    }
  }
  Float *data() {
    return ptr;
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
  // MatrixMap &matrix() { }
};

}

#endif
