#ifndef ocropus_batches__
#define ocropus_batches__

#include <array>
#include <vector>
#include "tensor.h"
#include "utils.h"

namespace ocropus {
using std::vector;

struct Batch {
  Tensor2 v;
  Tensor2 d;
  virtual ~Batch() {}
  int rows() const { return v.dimension(0); }
  int cols() const { return v.dimension(1); }
  int getGpu() { return v.getGpu(); }
  void clear() {
    v.setZero();
    d.setZero();
  }
  void zeroGrad() { d.setZero(rows(), cols()); }
};

struct BatchStorage : Batch {
  void setGpu(int n) {
    v.setGpu(n);
    d.setGpu(n);
  }
  void like(Batch &other) {
    setGpu(other.getGpu());
    resize(other.rows(), other.cols());
  }
  void setZero(int n, int m) {
    v.setZero(n, m);
    d.setZero(n, m);
  }
  void resize(int n, int m) { setZero(n, m); }
};

typedef BatchStorage Params;

// typedef vector<Mat> Sequence;
struct Sequence {
  int gpu = -1;
  vector<BatchStorage> steps;
  Float *data = nullptr;
  int dims[4] = {0, 0, 0, 0};

  TensorMap4 map4() {
    return TensorMap4(data, dims[0], dims[1], dims[2], dims[3]);
  }
  Sequence() {}
  Sequence(int N, int r, int b) { resize(N, r, b); }
  Sequence(Sequence &other) {
    like(other);
    copy(other);
  }
  Sequence(const Sequence &other) {
    like((Sequence &)other);
    copy((Sequence &)other);
  }
  ~Sequence() { free_gpu(data, gpu); }
  int getGpu() const { return gpu; }
  void setGpu(int n) {
    gpu = n;
    clear();
  }
  void clear() {
    steps.clear();
    if (data) free_gpu(data, gpu);
    data = nullptr;
    dims[0] = 0;
    dims[1] = 0;
    dims[2] = 0;
    dims[3] = 0;
  }
  void allocate(int N, int n, int m) {
    if (data) clear();
    dims[0] = n;
    dims[1] = m;
    dims[2] = 2;
    dims[3] = N;
    alloc_gpu(data, nbytes(), gpu);
  }

  int size() const { return dims[3]; }
  int rows() const { return dims[0]; }
  int cols() const { return dims[1]; }
  int total_size() const { return dims[0] * dims[1] * dims[2] * dims[3]; }
  int nbytes() const { return total_size() * sizeof *data; }
  void check() const {
    // the data pointer must be null iff the sequence has zero length
    assert(dims[3] == 0 ? !data : true);
    assert(!data ? dims[3] == 0 : true);
    if (!data) return;
    // batches must have non-zero size
    assert(steps[0].rows() > 0);
    assert(steps[0].cols() > 0);
    int N = size();
    int n = rows();
    int m = cols();
    for (int t = 0; t < N; t++) {
      // all batches must be displaced to the right locations and consistent
      assert(steps[t].v.displaced);
      assert(steps[t].d.displaced);
      assert(steps[t].v.ptr == data + (n * m) * (2 * t));
      assert(steps[t].d.ptr == data + (n * m) * (2 * t + 1));
      assert(steps[t].v.getGpu() == getGpu());
      assert(steps[t].rows() == steps[0].rows());
      assert(steps[t].cols() == steps[0].cols());
    }
  }
  void resize(int N, int n, int m) {
    check();
    if (N != size() || n != rows() || m != cols()) {
      clear();
      allocate(N, n, m);
      steps.resize(N);
      for (int t = 0; t < N; t++) {
        steps[t].v.displaceTo(data + (n * m) * (2 * t), n, m, gpu);
        steps[t].d.displaceTo(data + (n * m) * (2 * t + 1), n, m, gpu);
      }
    }
    //reset data, whether new or reused
    memset(data,0,nbytes());
  }
  void like(const Sequence &other) {
    resize(other.size(), other.rows(), other.cols());
  }

  void copy(const Sequence &other) {
    other.check();
    like(other);
    check();
    memcpy_gpu(data, gpu, other.data, other.gpu, nbytes());
  }
  void operator=(Sequence &other) { copy(other); }
  Batch &operator[](int i) { return steps[i]; }
  const Batch &operator[](int i) const { return steps[i]; }
  void zero() {
    for (int t = 0; t < steps.size(); t++) steps[t].clear();
  }
  void zeroGrad() {
    for (int t = 0; t < steps.size(); t++) steps[t].zeroGrad();
  }
};

void rinit(TensorMap2 m, Float s, const char *mode = "unif",
           Float offset = 0.0);
void rinit(Batch &m, int no, int ni, Float s, const char *mode = "unif",
           Float offset = 0.0);
void rinit(Params &m, int N, int no, int ni, Float s, const char *mode = "pos",
           Float offset = 0.0);
void rinit(Sequence &m, int no, int ni, Float s, const char *mode = "unif",
           Float offset = 0.0);
bool anynan(Batch &a);
bool anynan(Params &a);
bool anynan(Sequence &a);

inline void check_normalized(Batch &a) {
  for (int b = 0; b < a.cols(); b++) {
    double total = 0.0;
    for (int i = 0; i < a.rows(); i++) total += a.v(i, b);
    assert(fabs(total - 1.0) < 1e-5);
  }
}
inline void check_normalized(Sequence &a) {
  for (int t = 0; t < a.size(); t++) check_normalized(a[t]);
}
}

#endif
