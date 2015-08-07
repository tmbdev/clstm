#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {
using namespace std;
using Eigen::Ref;

namespace {
inline Float limexp(Float x) {
#if 1
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
#else
  return exp(x);
#endif
}

inline Float log_add(Float x, Float y) {
  if (abs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline Float log_mul(Float x, Float y) { return x + y; }
}

void forward_algorithm(Mat &lr, Mat &lmatch, double skip) {
  int n = ROWS(lmatch), m = COLS(lmatch);
  lr.resize(n, m);
  Vec v(m), w(m);
  for (int j = 0; j < m; j++) v(j) = skip * j;
  for (int i = 0; i < n; i++) {
    w.segment(1, m - 1) = v.segment(0, m - 1);
    w(0) = skip * i;
    for (int j = 0; j < m; j++) {
      Float same = log_mul(v(j), lmatch(i, j));
      Float next = log_mul(w(j), lmatch(i, j));
      v(j) = log_add(same, next);
    }
    lr.row(i) = v;
  }
}

void forwardbackward(Mat &both, Mat &lmatch) {
  Mat lr;
  forward_algorithm(lr, lmatch);
  Mat rlmatch = lmatch;
  rlmatch = rlmatch.rowwise().reverse().eval();
  rlmatch = rlmatch.colwise().reverse().eval();
  Mat rl;
  forward_algorithm(rl, rlmatch);
  rl = rl.colwise().reverse().eval();
  rl = rl.rowwise().reverse().eval();
  both = lr + rl;
}

void ctc_align_targets(Mat &posteriors, Mat &outputs, Mat &targets) {
  double lo = 1e-5;
  int n1 = ROWS(outputs);
  int n2 = ROWS(targets);
  int nc = COLS(targets);

  // compute log probability of state matches
  Mat lmatch;
  lmatch.resize(n1, n2);
  for (int t1 = 0; t1 < n1; t1++) {
    Vec out = outputs.row(t1);
    out = out.cwiseMax(lo);
    out /= out.sum();
    for (int t2 = 0; t2 < n2; t2++) {
      double value = out.transpose() * targets.row(t2).transpose();
      lmatch(t1, t2) = log(value);
    }
  }
  // compute unnormalized forward backward algorithm
  Mat both;
  forwardbackward(both, lmatch);

  // compute normalized state probabilities
  Mat epath = (both.array() - both.maxCoeff()).unaryExpr(ptr_fun(limexp));
  for (int j = 0; j < n2; j++) {
    double l = epath.col(j).sum();
    epath.col(j) /= l == 0 ? 1e-9 : l;
  }
  debugmat = epath;

  // compute posterior probabilities for each class and normalize
  Mat aligned;
  aligned.resize(n1, nc);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < nc; j++) {
      double total = 0.0;
      for (int k = 0; k < n2; k++) {
        double value = epath(i, k) * targets(k, j);
        total += value;
      }
      aligned(i, j) = total;
    }
  }
  for (int i = 0; i < n1; i++) {
    aligned.row(i) /= fmax(1e-9, aligned.row(i).sum());
  }

  posteriors = aligned;
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Sequence &targets) {
  assert(COLS(outputs[0]) == 1);
  assert(COLS(targets[0]) == 1);
  int n1 = outputs.size();
  int n2 = targets.size();
  int nc = targets[0].size();
  Mat moutputs(n1, nc);
  Mat mtargets(n2, nc);
  for (int i = 0; i < n1; i++) moutputs.row(i) = outputs[i].col(0);
  for (int i = 0; i < n2; i++) mtargets.row(i) = targets[i].col(0);
  Mat aligned;
  ctc_align_targets(aligned, moutputs, mtargets);
  posteriors.resize(n1);
  for (int i = 0; i < n1; i++) {
    posteriors[i].resize(aligned.row(i).size(), 1);
    posteriors[i].col(0) = aligned.row(i);
  }
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Classes &targets) {
  int nclasses = outputs[0].size();
  Sequence stargets;
  stargets.resize(targets.size());
  for (int t = 0; t < stargets.size(); t++) {
    stargets[t].resize(nclasses, 1);
    stargets[t].fill(0);
    stargets[t](targets[t], 0) = 1.0;
  }
  ctc_align_targets(posteriors, outputs, stargets);
}

void mktargets(Sequence &seq, Classes &transcript, int ndim) {
  seq.resize(2 * transcript.size() + 1);
  for (int t = 0; t < seq.size(); t++) {
    seq[t].setZero(ndim, 1);
    if (t % 2 == 1)
      seq[t](transcript[(t - 1) / 2]) = 1;
    else
      seq[t](0) = 1;
  }
}

void trivial_decode(Classes &cs, Sequence &outputs, int batch,
                    vector<int> *locs) {
  cs.clear();
  if (locs) locs->clear();
  int N = outputs.size();
  int t = 0;
  float mv = 0;
  int mc = -1;
  int mt = -1;
  while (t < N) {
    int index;
    float v = outputs[t].col(batch).maxCoeff(&index);
    if (index == 0) {
      // NB: there should be a 0 at the end anyway
      if (mc != -1 && mc != 0) {
        cs.push_back(mc);
        if (locs) locs->push_back(mt);
      }
      mv = 0;
      mc = -1;
      mt = -1;
      t++;
      continue;
    }
    if (v > mv) {
      mv = v;
      mc = index;
      mt = t;
    }
    t++;
  }
}

void trivial_decode(Classes &cs, Sequence &outputs, int batch) {
  trivial_decode(cs, outputs, batch, nullptr);
}
}  // ocropus
