#ifndef clstm_compute__
#define clstm_compute__

#include "utils.h"
#include "batches.h"

namespace ocropus {
using namespace std;

template <class RHS>
auto operator>>(Batch &b, RHS rhs) -> decltype(b.v >> rhs) {
  return b.v >> rhs;
}

struct NoNonlin {
  static constexpr const char *kind = "Linear";
  static inline Float nonlin(Float x) { return x; }
  static inline Float yderiv(Float y) { return 1; }
};
struct SigmoidNonlin {
  static constexpr const char *kind = "Sigmoid";
  static inline Float nonlin(Float x) { return sigmoid(x); }
  static inline Float yderiv(Float y) { return y * (1 - y); }
};
struct TanhNonlin {
  static constexpr const char *kind = "Tanh";
  static inline Float nonlin(Float x) { return tanh(x); }
  static inline Float yderiv(Float y) { return 1 - y * y; }
};
struct ReluNonlin {
  static constexpr const char *kind = "Relu";
  static inline Float nonlin(Float x) { return relu_(x); }
  static inline Float yderiv(Float y) { return heavi_(y); }
};

void forward_stack(Batch &z, Batch &x, Batch &y);
void backward_stack(Batch &z, Batch &x, Batch &y);
void forward_stack(Batch &z, Batch &x, Sequence &y, int last);
void backward_stack(Batch &z, Batch &x, Sequence &y, int last);

void forward_reverse(Sequence &y, Sequence &x);
void backward_reverse(Sequence &y, Sequence &x);

template <class F>
void forward_full1(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full1(Batch &y, Params &W, Batch &x);

void forward_softmax(Batch &z, Params &W1, Batch &x);
void backward_softmax(Batch &z, Params &W1, Batch &x);
void forward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);
void backward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf);
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf);
template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go);
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go);

#ifdef DEPRECATED
void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
template <class F>
void forward_full(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full(Batch &y, Params &W, Batch &x);
#endif

void rinit(Batch &m, int no, int ni, Float s, const string mode = "unif", Float offset=0.0);
void rinit(Sequence &m, int no, int ni, Float s, const string mode = "unif", Float offset=0.0);
void rinit(Params &m, int N, int no, int ni, Float s, const string mode = "pos", Float offset=0.0);
bool anynan(Batch &a);
bool anynan(Sequence &a);
bool anynan(Params &a);

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
  a = temp;
}
}

#endif
