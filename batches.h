#ifndef ocropus_batches__
#define ocropus_batches__

#include <vector>
#include "tensor.h"

namespace ocropus {
using std::vector;

struct Batch {
  Tensor2 v;
  Tensor2 d;
  int rows() const { return v.dimension(0); }
  int cols() const { return v.dimension(1); }
  void setGpu(int n) {
    v.setGpu(n);
    d.setGpu(n);
  }
  int getGpu() {
    return v.getGpu();
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
  void clear() {
    v.setZero();
    d.setZero();
  }
  void zeroGrad() { d.setZero(rows(), cols()); }
  void gradientClip(Float clip) {
    TensorMap2 d = *this->d;
    if (clip>=1e6) return;
    assert(clip>0);
    for(int i=0; i<rows(); i++) {
      for(int j=0; j<cols(); j++) {
        d(i,j) = fmax(-clip, fmin(clip, d(i,j)));
      }
    }
  }
};

typedef Batch Params;

// typedef vector<Mat> Sequence;
struct Sequence {
  vector<Batch> steps;
  Sequence() {}
  Sequence(int N, int r, int b) { resize(N, r, b); }
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
  int size() const { return steps.size(); }
  void resize(int N, int n, int m) {
    steps.resize(N);
    for (int t = 0; t < N; t++) {
      steps[t].resize(n, m);
    }
  }
  void like(const Sequence &other) {
    resize(other.size(), other.rows(), other.cols());
  }
  void copy(const Sequence &other) {
    like(other);
    for (int t = 0; t < other.size(); t++) steps[t] = other[t];
  }
  Batch &operator[](int i) { return steps[i]; }
  const Batch &operator[](int i) const { return steps[i]; }
  void zero() {
    for (int t = 0; t < steps.size(); t++) steps[t].clear();
  }
  void zeroGrad() {
    for (int t = 0; t < steps.size(); t++) steps[t].zeroGrad();
  }
  void gradientClip(Float gc) {
    for(int i=0;i<steps.size(); i++)
      steps[i].gradientClip(gc);
  }
};

void rinit(TensorMap2 m, Float s, const char *mode = "unif", Float offset=0.0);
void rinit(Batch &m, int no, int ni, Float s, const char *mode = "unif", Float offset=0.0);
void rinit(Params &m, int N, int no, int ni, Float s, const char *mode = "pos", Float offset=0.0);
void rinit(Sequence &m, int no, int ni, Float s, const char *mode = "unif", Float offset=0.0);
bool anynan(Batch &a);
bool anynan(Params &a);
bool anynan(Sequence &a);

}

#endif
