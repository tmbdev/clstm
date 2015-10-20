#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>
#include "clstm_compute.h"

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {
using namespace std;
using Eigen::Ref;

static void forward_algorithm(Tensor2 &lr, Tensor2 &lmatch, double skip = -5) {
  int n = rows(lmatch), m = cols(lmatch);
  lr.resize(n, m);
  Tensor1 v(m), w(m);
  for (int j = 0; j < m; j++) v(j) = skip * j;
  for (int i = 0; i < n; i++) {
    w(0) = skip * i;
    for (int j = 1; j < m; j++) w(j) = v(j - 1);
    for (int j = 0; j < m; j++) {
      Float same = log_mul(v(j), lmatch(i, j));
      Float next = log_mul(w(j), lmatch(i, j));
      v(j) = log_add(same, next);
    }
    for (int j = 0; j < m; j++) lr(i, j) = v(j);
  }
}

static void forwardbackward(Tensor2 &both, Tensor2 &lmatch) {
  int n = rows(lmatch), m = cols(lmatch);
  Tensor2 lr;
  forward_algorithm(lr, lmatch);
  Tensor2 rlmatch(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) rlmatch(i, j) = lmatch(n - i - 1, m - j - 1);
  Tensor2 rrl;
  forward_algorithm(rrl, rlmatch);
  Tensor2 rl(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) rl(i, j) = rrl(n - i - 1, m - j - 1);
  both = lr + rl;
}

void ctc_align_targets(Tensor2 &posteriors, Tensor2 &outputs,
                       Tensor2 &targets) {
  double lo = 1e-5;
  int n1 = rows(outputs);
  int n2 = rows(targets);
  int nc = cols(targets);
  assert(nc == cols(outputs));

  // compute log probability of state matches
  Tensor2 lmatch;
  lmatch.resize(n1, n2);
  for (int t1 = 0; t1 < n1; t1++) {
    Tensor1 out(nc);
    for (int i = 0; i < nc; i++) out(i) = fmax(lo, outputs(t1, i));
    out = out / sum(out);
    for (int t2 = 0; t2 < n2; t2++) {
      double total = 0.0;
      for (int k = 0; k < nc; k++) total += out(k) * targets(t2, k);
      lmatch(t1, t2) = log(total);
    }
  }
  // compute unnormalized forward backward algorithm
  Tensor2 both;
  forwardbackward(both, lmatch);

  // compute normalized state probabilities
  Tensor2 epath = (both - maximum(both)).unaryExpr(ptr_fun(limexp));
  for (int j = 0; j < n2; j++) {
    double total = 0.0;
    for (int i = 0; i < rows(epath); i++) total += epath(i, j);
    total = fmax(1e-9, total);
    for (int i = 0; i < rows(epath); i++) epath(i, j) /= total;
  }

  // compute posterior probabilities for each class and normalize
  Tensor2 aligned;
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
    double total = 0.0;
    for (int j = 0; j < nc; j++) total += aligned(i, j);
    total = fmax(total, 1e-9);
    for (int j = 0; j < nc; j++) aligned(i, j) /= total;
  }

  posteriors = aligned;
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Sequence &targets) {
  assert(outputs.cols() == 1);
  assert(targets.cols() == 1);
  assert(outputs.rows() == targets.rows());
  int n1 = outputs.size();
  int n2 = targets.size();
  int nc = targets[0].rows();
  Tensor2 moutputs(n1, nc);
  Tensor2 mtargets(n2, nc);
  for (int i = 0; i < n1; i++)
    for (int j = 0; j < nc; j++) moutputs(i, j) = outputs[i].v(j, 0);
  for (int i = 0; i < n2; i++)
    for (int j = 0; j < nc; j++) mtargets(i, j) = targets[i].v(j, 0);
  Tensor2 aligned;
  ctc_align_targets(aligned, moutputs, mtargets);
  posteriors.resize(n1, nc, 1);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < nc; j++) posteriors[i].v(j, 0) = aligned(i, j);
  }
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Classes &targets) {
  int nclasses = outputs[0].v.size();
  Sequence stargets;
  stargets.resize(targets.size());
  for (int t = 0; t < stargets.size(); t++) {
    stargets[t].resize(nclasses, 1);
    stargets[t].v.fill(0);
    stargets[t].v(targets[t], 0) = 1.0;
  }
  ctc_align_targets(posteriors, outputs, stargets);
}

void mktargets(Sequence &seq, Classes &transcript, int ndim) {
  seq.resize(2 * transcript.size() + 1);
  for (int t = 0; t < seq.size(); t++) {
    seq[t].setZero(ndim, 1);
    if (t % 2 == 1)
      seq[t].v(transcript[(t - 1) / 2]) = 1;
    else
      seq[t].v(0) = 1;
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
    int index = argmax(outputs[t].v.col(batch));
    float v = outputs[t].v(index, batch);
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
