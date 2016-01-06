// -*- C++ -*-

// Additional functions and utilities for CLSTM networks.
// These may use the array classes from "multidim.h"

#ifndef ocropus_clstm_extras_
#define ocropus_clstm_extras_

#include <glob.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <map>
#include <string>
#include "clstm.h"
#include "pstring.h"

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

// text line normalization

struct INormalizer {
  int target_height = 48;
  float smooth2d = 1.0;
  float smooth1d = 0.3;
  float range = 4.0;
  float vscale = 1.0;
  virtual ~INormalizer() {}
  virtual void getparams(bool verbose = false) {}
  virtual void measure(TensorMap2 line) = 0;
  virtual void normalize(Tensor2 &out, TensorMap2 in) = 0;
  virtual void setPyServer(void *p) {}
};

INormalizer *make_Normalizer(const string &);
INormalizer *make_NoNormalizer();
INormalizer *make_MeanNormalizer();
INormalizer *make_CenterNormalizer();

void read_png(Tensor2 &image, const char *name);
void write_png(const char *name, TensorMap2 image);
}

#endif
