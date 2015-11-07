#ifndef clstm_compute__
#define clstm_compute__

#include "utils.h"
#include "batches.h"

namespace ocropus {
using namespace std;

enum { LIN = 0, SIG = 1, TANH = 2, RELU = 3 };
typedef int Nonlin;

#define DEFGENERIC(NAME, ARGS) void NAME ARGS

DEFGENERIC(forward_stack,(Batch &z, Batch &x, Batch &y));
DEFGENERIC(backward_stack,(Batch &z, Batch &x, Batch &y));
DEFGENERIC(forward_stack,(Batch &z, Batch &x, Sequence &y, int last));
DEFGENERIC(backward_stack,(Batch &z, Batch &x, Sequence &y, int last));
DEFGENERIC(forward_reverse,(Sequence &y, Sequence &x));
DEFGENERIC(backward_reverse,(Sequence &y, Sequence &x));
DEFGENERIC(forward_full1,(Batch &y, Params &W, Batch &x, Nonlin nl));
DEFGENERIC(backward_full1,(Batch &y, Params &W, Batch &x, Nonlin nl));
DEFGENERIC(forward_softmax,(Batch &z, Params &W1, Batch &x));
DEFGENERIC(backward_softmax,(Batch &z, Params &W1, Batch &x));
DEFGENERIC(forward_softmax,(Sequence &outputs, Params &W1, Sequence &inputs));
DEFGENERIC(backward_softmax,(Sequence &outputs, Params &W1, Sequence &inputs));
DEFGENERIC(forward_statemem,(Batch &state, Batch &ci, Batch &gi, Sequence &states, int last, Batch &gf));
DEFGENERIC(backward_statemem,(Batch &state, Batch &ci, Batch &gi, Sequence &states, int last, Batch &gf));
DEFGENERIC(forward_nonlingate,(Batch &out, Batch &state, Batch &go, Nonlin nl));
DEFGENERIC(backward_nonlingate,(Batch &out, Batch &state, Batch &go, Nonlin nl));

}

#endif
