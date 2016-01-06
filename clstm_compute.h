#ifndef clstm_compute__
#define clstm_compute__

#include <utility>
#include "batches.h"

namespace ocropus {
using namespace std;

constexpr int LIN = 0;
constexpr int SIG = 1;
constexpr int TANH = 2;
constexpr int RELU = 3;
constexpr int LOGMAG = 4;

extern Eigen::DefaultDevice default_device;

inline int gpu_id(Tensor2 &t) { return t.getGpu(); }
inline int gpu_id(Batch &b) { return gpu_id(b.v); }
inline int gpu_id(Sequence &s) { return gpu_id(s[0]); }

// If this has been compiled with CUDA, there is a gpu_device
// function in the CUDA-compiled code; otherwise, we default
// to something that always returns a nullptr for the GPU
// device.

#ifdef CLSTM_CUDA
Eigen::GpuDevice *gpu_device(int id);
#else
inline Eigen::GpuDevice *gpu_device(int id) {
  assert(id < 0);
  return nullptr;
}
#endif

template <class T>
inline Eigen::GpuDevice *gpu(T arg) {
  int id = gpu_id(arg);
  return gpu_device(id);
}

// This bit of macro and template magic allows us to
// transparently select between CPU and GPU versions of
// computations. The computations themselves are
// expressed using standard Eigen::Tensor notation and
// devices in clstm_compute.cc. Only clstm_compute.cc
// needs to be compiled with nvcc, greatly cutting down
// on the exposure to incompatibilities and bugs in nvcc.

#ifdef CLSTM_CUDA
#define DEFGENERIC(NAME, ...)                                \
  template <typename Arg, typename... Args>                  \
  void NAME(Arg &&arg, Args &&... args) {                    \
    extern void NAME(Eigen::DefaultDevice *, __VA_ARGS__);   \
    extern void NAME(Eigen::GpuDevice *, __VA_ARGS__);       \
    Eigen::GpuDevice *dev = gpu_device(gpu_id(arg));         \
    if (dev) {                                               \
      NAME(dev, arg, std::forward<Args>(args)...);           \
      return;                                                \
    }                                                        \
    NAME(&default_device, arg, std::forward<Args>(args)...); \
  }
#else
#define DEFGENERIC(NAME, ...)                                \
  template <typename Arg, typename... Args>                  \
  void NAME(Arg &&arg, Args &&... args) {                    \
    extern void NAME(Eigen::DefaultDevice *, __VA_ARGS__);   \
    NAME(&default_device, arg, std::forward<Args>(args)...); \
  }
#endif

DEFGENERIC(forward_nonlin, Batch &, Batch &, int);
DEFGENERIC(backward_nonlin, Batch &, Batch &, int);
DEFGENERIC(forward_nonlin0, Batch &, int);
DEFGENERIC(backward_nonlin0, Batch &, int);
DEFGENERIC(forward_lin1, Batch &, Params &, Batch &);
DEFGENERIC(backward_lin1, Batch &, Params &, Batch &);
DEFGENERIC(forward_full1, Batch &, Params &, Batch &, int);
DEFGENERIC(backward_full1, Batch &, Params &, Batch &, int);
DEFGENERIC(forward_stack, Batch &, Batch &, Batch &);
DEFGENERIC(backward_stack, Batch &, Batch &, Batch &);
DEFGENERIC(forward_stack_delay, Batch &, Batch &, Sequence &, int);
DEFGENERIC(backward_stack_delay, Batch &, Batch &, Sequence &, int);
DEFGENERIC(forward_reverse, Sequence &, Sequence &);
DEFGENERIC(backward_reverse, Sequence &, Sequence &);
DEFGENERIC(forward_btswitch, Sequence &, Sequence &);
DEFGENERIC(backward_btswitch, Sequence &, Sequence &);
DEFGENERIC(forward_batchstack, Sequence &, Sequence &, int pre = 1,
           int post = 1);
DEFGENERIC(backward_batchstack, Sequence &, Sequence &, int pre = 1,
           int post = 1);
DEFGENERIC(forward_softmax, Batch &, Params &, Batch &);
DEFGENERIC(backward_softmax, Batch &, Params &, Batch &);
DEFGENERIC(forward_statemem, Batch &, Batch &, Batch &, Sequence &, int,
           Batch &);
DEFGENERIC(backward_statemem, Batch &, Batch &, Batch &, Sequence &, int,
           Batch &);
DEFGENERIC(forward_nonlingate, Batch &, Batch &, Batch &, int);
DEFGENERIC(backward_nonlingate, Batch &, Batch &, Batch &, int);

DEFGENERIC(fill, Tensor2 &, Float value);
DEFGENERIC(clip_gradient, Batch &, Float value);
DEFGENERIC(sgd_update, Params &, Float lr, Float mom);
};

#endif
