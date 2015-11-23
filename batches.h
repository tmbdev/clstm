#ifndef ocropus_batches__
#define ocropus_batches__

#include <vector>
#include <array>
#include "tensor.h"

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
  int dims[4] = {0,0,0,0};
  Sequence() {
  }
  Sequence(int N, int r, int b) {
    resize(N, r, b);
  }
  int getGpu() { return gpu; }
  void setGpu(int n) {
    gpu = n;
    clear();
  }
  void clear() {
    steps.clear();
    if(data) free(data);
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
    data = (Float*)malloc(total_size() * sizeof *data);
  }

  int size() const { return dims[3]; }
  int rows() const { return dims[0]; }
  int cols() const { return dims[1]; }
  int total_size() const { return dims[0] * dims[1] * dims[2] * dims[3]; }
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
  void resize(int N, int n, int m) {
    if (N==size() && n==rows() && m==cols()) {
      for (int t = 0; t < N; t++) {
	steps[t].v.setZero();
	steps[t].d.setZero();
      }
    } else {
      clear();
      allocate(N, n, m);
      steps.resize(N);
      for (int t = 0; t < N; t++) {
#if 0
	if (steps[t].getGpu() != gpu) steps[t].setGpu(gpu);
	steps[t].resize(n, m);
	steps[t].v.resizeable = false;
	steps[t].d.resizeable = false;
#else
	steps[t].v.displaceTo(data + (n*m)*(2*t), n, m, gpu);
	steps[t].d.displaceTo(data + (n*m)*(2*t+1), n, m, gpu);
#endif
      }
    }
  }
  void like(const Sequence &other) {
    // don't assign GPU status
    resize(other.size(), other.rows(), other.cols());
  }
  void copy(const Sequence &other) {
    // don't assign GPU status
    like(other);
    for (int t = 0; t < other.size(); t++) steps[t] = other.steps[t];
  }
  void operator=(Sequence &other) {
    // don't assign GPU status
    copy(other);
  }
  Batch &operator[](int i) {
    return steps[i];
  }
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
}

#endif
