#include "batches.h"
#include <string>

// random initialization of sequences etc.

namespace {

// very simple "random" number generator; this
// is just used for initializations

double state = getenv("seed") ? atof(getenv("seed")) : 0.1;

inline double randu() {
  state = 189843.9384938 * state + 0.328340981343;
  state -= floor(state);
  return state;
}

inline double randn() {
  double u1 = randu();
  double u2 = randu();
  double r = -2 * log(u1);
  double theta = 2 * M_PI * u2;
  double z0 = r * cos(theta);
  return z0;
}
}

namespace ocropus {

// Random initializations with different distributions.

void rinit(TensorMap2 a, Float s, const char *mode_, Float offset) {
  int n = a.dimension(0), m = a.dimension(1);
  std::string mode(mode_);
  if (mode == "unif") {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a(i, j) = 2 * s * randu() - s + offset;
  } else if (mode == "negbiased") {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a(i, j) = 3 * s * randu() - 2 * s + offset;
  } else if (mode == "pos") {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a(i, j) = s * randu() + offset;
  } else if (mode == "neg") {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a(i, j) = -s * randu() + offset;
  } else if (mode == "normal") {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a(i, j) = s * randn() + offset;
  }
}

void rinit(Tensor2 &t, int r, int c, Float s, const char *mode_, Float offset) {
  // use a temporary so that initialization of GPU tensors works
  Tensor2 temp;
  temp.resize(r, c);
  rinit(temp(), s, mode_, offset);
  t = temp;
}

void rinit(Batch &m, int r, int c, Float s, const char *mode, Float offset) {
  rinit(m.v, r, c, s, mode, offset);
  m.zeroGrad();
}

void rinit(Sequence &m, int N, int r, int c, Float s, const char *mode,
           Float offset) {
  m.steps.resize(N);
  for (int t = 0; t < N; t++) rinit(m[t], r, c, s, mode, offset);
}

// checking for NaNs in different objects

bool anynan(TensorMap2 a) {
  for (int j = 0; j < a.dimension(0); j++) {
    for (int k = 0; k < a.dimension(1); k++) {
      float x = a(j, k);
      if (std::isnan(x)) return true;
    }
  }
  return false;
}

bool anynan(Batch &a) {
  if(anynan(a.v())) return true;
  if(anynan(a.d())) return true;
  return false;
}
bool anynan(Params &a) {
  if (anynan(a.v())) return true;
  if (anynan(a.d())) return true;
  return false;
}

bool anynan(Sequence &a) {
  for (int i = 0; i < a.size(); i++)
    if (anynan(a[i])) return true;
  return false;
}
}
