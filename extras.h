// -*- C++ -*-

// Additional functions and utilities for CLSTM networks.
// These may use the array classes from "multidim.h"

#ifndef ocropus_clstm_extras_
#define ocropus_clstm_extras_

#include "clstm.h"
#include <string>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <map>
#include <stdarg.h>
#include <glob.h>
#include "pstring.h"
#include <iostream>

namespace ocropus {
using std::string;
using std::wstring;
using std::shared_ptr;
using std::vector;
using std::cout;
using std::ostream;
using std::cerr;
using std::endl;
using std::min;

void glob(vector<string> &result, const string &arg);

// text line normalization

struct INormalizer {
  int target_height = 48;
  float smooth2d = 1.0;
  float smooth1d = 0.3;
  float range = 4.0;
  float vscale = 1.0;
  virtual ~INormalizer() {}
  virtual void getparams(bool verbose = false) {}
  virtual void measure(Tensor<float,2> &line) = 0;
  virtual void normalize(Tensor<float,2> &out, Tensor<float,2> &in) = 0;
  virtual void setPyServer(void *p) {}
};

INormalizer *make_Normalizer(const string &);
INormalizer *make_NoNormalizer();
INormalizer *make_MeanNormalizer();
INormalizer *make_CenterNormalizer();

void read_png(Tensor<unsigned char,3> &image, FILE *fp);
void write_png(FILE *fp, Tensor<float,3> &image);
void read_png(Tensor<float,2> &image, const char *name);
void write_png(const char *name, Tensor<float,2> &image);

inline bool anynan(Tensor<float,1> &a) {
  for (int i = 0; i < a.size(); i++)
    if (isnan(a[i])) return true;
  return false;
}
inline bool anynan(Tensor<float, 2> &a) {
  for (int i = 0; i < a.dimension(0); i++)
    for (int j = 0; j < a.dimension(0); j++)
      if (isnan(a(i, j))) return true;
  return false;
}


inline void assign(Tensor<int,1> &dest, vector<int> &src) {
  int n = src.size();
  dest.resize(n);
  for (int i = 0; i < n; i++) dest[i] = src[i];
}

inline void assign(vector<int> &dest, Tensor<int,1> &src) {
  int n = src.dimension(0);
  dest.resize(n);
  for (int i = 0; i < n; i++) dest[i] = src[i];
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
  assign(a, temp);
}

template <class T>
inline void assign(Sequence &seq, T &a) {
  assert(a.rank() == 2);
  seq.resize(a.dimension(0));
  for (int t = 0; t < a.dimension(0); t++) {
    seq[t].resize(a.dimension(1), 1);
    for (int i = 0; i < a.dimension(1); i++) seq[t].v(i, 0) = a(t, i);
  }
}

#if FIXME
template <class T>
inline void assign(T &a, Sequence &seq) {
  a.resize(int(seq.size()), int(seq[0].v.size()));
  for (int t = 0; t < a.dimension(0); t++) {
    for (int i = 0; i < a.dimension(1); i++) a(t, i) = seq[t].v(i);
  }
}
#endif

template <class A, class T>
inline int indexof(A &a, const T &t) {
  for (int i = 0; i < a.size(); i++)
    if (a[i] == t) return i;
  return -1;
}

// simple network creation; this takes parameters from the environment
Network make_net_init(const string &kind, int nclasses, int dim,
                      string prefix = "");

// setting inputs and outputs
void set_inputs(INetwork *net, Tensor<float,2> &inputs);
void set_targets(INetwork *net, Tensor<float,2> &targets);
void set_targets_accelerated(INetwork *net, Tensor<float,2> &targets);
void set_classes(INetwork *net, Tensor<int,1> &targets);

// single sequence training functions
void mktargets(Tensor<float,2> &seq, Tensor<int,1> &targets, int ndim);
void train(INetwork *net, Tensor<float,2> &inputs, Tensor<float,2> &targets);
void ctrain(INetwork *net, Tensor<float,2> &inputs, Tensor<int,1> &targets);
void cpred(INetwork *net, Tensor<int,1> &preds, Tensor<float,2> &inputs);
void ctc_train(INetwork *net, Tensor<float,2> &xs, Tensor<float,2> &targets);
void ctc_train(INetwork *net, Tensor<float,2> &xs, Tensor<int,1> &targets);
void ctc_train(INetwork *net, Tensor<float,2> &xs, string &targets);
void ctc_train(INetwork *net, Tensor<float,2> &xs, wstring &targets);
}

#endif
