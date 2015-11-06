#ifndef clstm_compute__
#define clstm_compute__

#include "utils.h"
#include "batches.h"

namespace ocropus {
using namespace std;

enum { LIN = 0, SIG = 1, TANH = 2, RELU = 3 };
typedef int Nonlin;

void forward_stack(Batch &z, Batch &x, Batch &y);
void backward_stack(Batch &z, Batch &x, Batch &y);
void forward_stack(Batch &z, Batch &x, Sequence &y, int last);
void backward_stack(Batch &z, Batch &x, Sequence &y, int last);

void forward_reverse(Sequence &y, Sequence &x);
void backward_reverse(Sequence &y, Sequence &x);

void forward_full1(Batch &y, Params &W, Batch &x, Nonlin nl);
void backward_full1(Batch &y, Params &W, Batch &x, Nonlin nl);

void forward_softmax(Batch &z, Params &W1, Batch &x);
void backward_softmax(Batch &z, Params &W1, Batch &x);
void forward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);
void backward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf);
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf);
void forward_nonlingate(Batch &out, Batch &state, Batch &go, Nonlin nl);
void backward_nonlingate(Batch &out, Batch &state, Batch &go, Nonlin nl);

#ifdef DEPRECATED
void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
template <class F>
void forward_full(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full(Batch &y, Params &W, Batch &x);
#endif

}

#endif
