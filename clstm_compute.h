#ifndef clstm_compute__
#define clstm_compute__

#include "utils.h"
#include "batches.h"

namespace ocropus {
using namespace std;

enum { LIN = 0, SIG = 1, TANH = 2, RELU = 3 };
typedef int Nonlin;

// DEFGENERIC(NAME, arglist)
// DEFMETHOD(NAME)(arglist) {body}

#if 0
#define DEFGENERIC(NAME, ARGS) void NAME ARGS
#define DEFMETHOD(NAME) void NAME
#define BEGINMETHODS
#define ENDMETHODS
#else
#define DEFGENERIC(NAME, ARGS) void (*NAME) ARGS;
#define DEFMETHOD(NAME) ; NAME = [] /* (args) { body } */
#define BEGINMETHODS static int _dummy = [] {
#define ENDMETHODS ; return 0; } ();
#endif

DEFGENERIC(forward_stack,(Batch &, Batch &, Batch &));
DEFGENERIC(backward_stack,(Batch &, Batch &, Batch &));
DEFGENERIC(forward_stack_delay,(Batch &, Batch &, Sequence &, int));
DEFGENERIC(backward_stack_delay,(Batch &, Batch &, Sequence &, int));
DEFGENERIC(forward_reverse,(Sequence &, Sequence &));
DEFGENERIC(backward_reverse,(Sequence &, Sequence &));
DEFGENERIC(forward_full1,(Batch &, Params &, Batch &, Nonlin));
DEFGENERIC(backward_full1,(Batch &, Params &, Batch &, Nonlin));
DEFGENERIC(forward_softmax,(Batch &, Params &, Batch &));
DEFGENERIC(backward_softmax,(Batch &, Params &, Batch &));
DEFGENERIC(forward_statemem,(Batch &, Batch &, Batch &gi, Sequence &, int, Batch &));
DEFGENERIC(backward_statemem,(Batch &, Batch &, Batch &gi, Sequence &, int, Batch &));
DEFGENERIC(forward_nonlingate,(Batch &, Batch &, Batch &, int));
DEFGENERIC(backward_nonlingate,(Batch &, Batch &, Batch &, int));

}

#endif
