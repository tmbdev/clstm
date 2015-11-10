#ifndef clstm_compute__
#define clstm_compute__

#include "batches.h"
#include <utility>

namespace ocropus {
using namespace std;

enum { LIN = 0, SIG = 1, TANH = 2, RELU = 3 };
typedef int Nonlin;

extern Eigen::DefaultDevice default_device;

// This bit of macro and template magic allows us to
// transparently select between CPU and GPU versions of
// computations. The computations themselves are
// expressed using standard Eigen::Tensor notation and
// devices in clstm_compute.cc. Only clstm_compute.cc
// needs to be compiled with nvcc, greatly cutting down
// on the exposure to incompatibilities and bugs in nvcc.

#define DEFGENERIC(NAME, ...) \
template <typename... Args> \
void NAME(Args&&... args) { \
  extern void NAME(Eigen::DefaultDevice *,__VA_ARGS__); \
  NAME(&default_device, std::forward<Args>(args)...); \
}

DEFGENERIC(forward_stack, Batch &, Batch &, Batch &);
DEFGENERIC(backward_stack, Batch &, Batch &, Batch &);
DEFGENERIC(forward_stack_delay, Batch &, Batch &, Sequence &, int);
DEFGENERIC(backward_stack_delay, Batch &, Batch &, Sequence &, int);
DEFGENERIC(forward_reverse, Sequence &, Sequence &);
DEFGENERIC(backward_reverse, Sequence &, Sequence &);
DEFGENERIC(forward_full1, Batch &, Params &, Batch &, Nonlin);
DEFGENERIC(backward_full1, Batch &, Params &, Batch &, Nonlin);
DEFGENERIC(forward_softmax, Batch &, Params &, Batch &);
DEFGENERIC(backward_softmax, Batch &, Params &, Batch &);
DEFGENERIC(forward_statemem, Batch &, Batch &, Batch &gi, Sequence &, int, Batch &);
DEFGENERIC(backward_statemem, Batch &, Batch &, Batch &gi, Sequence &, int, Batch &);
DEFGENERIC(forward_nonlingate, Batch &, Batch &, Batch &, int);
DEFGENERIC(backward_nonlingate, Batch &, Batch &, Batch &, int);

DEFGENERIC(sgd_update, Params &,Float lr, Float mom);
};

#endif
