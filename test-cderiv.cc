#include <assert.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "clstm.h"
#include "clstm_compute.h"
#include "extras.h"
#include "utils.h"

using std_string = std::string;
#define string std_string
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::to_string;
using std::make_pair;
using std::cout;
using std::stoi;
using namespace Eigen;
using namespace ocropus;

typedef vector<Params> ParamVec;

double sqr(double x) { return x * x; }

double randu() {
  static int count = 1;
  for (;;) {
    double x = cos(count * 3.7);
    count++;
    if (fabs(x) > 0.1) return x;
  }
}

void randseq(Sequence &a, int N, int n, int m) {
  bool finit = getienv("finit", 0);
  a.resize(N, n, m);
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        if (finit) {
          a[t].v(i, j) = 10000 * t + 100 * i + j;
          a[t].d(i, j) = 10000 * t + 100 * i + j + 0.5;
        } else {
          a[t].v(i, j) = randu();
          a[t].d(i, j) = randu();
        }
      }
    }
  }
  a.check();
}

void randparams(ParamVec &a, const vector<vector<int>> &specs) {
  int N = specs.size();
  a.resize(N);
  for (int k = 0; k < N; k++) {
    int n = specs[k][0];
    int m = specs[k][1];
    a[k].setZero(n, m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        a[k].v(i, j) = randu();
        a[k].d(i, j) = randu();
      }
    }
  }
}

double maxerr(Sequence &out, Sequence &target) {
  assert(out.size() == target.size());
  assert(out.rows() == target.rows());
  assert(out.cols() == target.cols());
  int N = out.size(), n = out.rows(), m = out.cols();
  double maxerr = 0.0;
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        double delta = target[t].v(i, j) - out[t].v(i, j);
        if (fabs(delta) > maxerr) maxerr = fabs(delta);
      }
    }
  }
  return maxerr;
}

double avgerr(Sequence &out, Sequence &target) {
  assert(out.size() == target.size());
  assert(out.rows() == target.rows());
  assert(out.cols() == target.cols());
  int N = out.size(), n = out.rows(), m = out.cols();
  double total = 0.0;
  int count = 0;
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        double delta = target[t].v(i, j) - out[t].v(i, j);
        total += fabs(delta);
        count++;
      }
    }
  }
  return total / count;
}

double mse(Sequence &out, Sequence &target) {
  assert(out.size() == target.size());
  assert(out.rows() == target.rows());
  assert(out.cols() == target.cols());
  int N = out.size(), n = out.rows(), m = out.cols();
  double total = 0.0;
  for (int t = 0; t < N; t++) {
    out[t].zeroGrad();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        double delta = target[t].v(i, j) - out[t].v(i, j);
        out[t].d(i, j) = delta;
        total += sqr(delta);
      }
    }
  }
  return total;
}

struct Minimizer {
  double value = INFINITY;
  double param = 0;
  void add(double value, double param = NAN) {
    if (value >= this->value) return;
    this->value = value;
    this->param = param;
  }
};

struct Maximizer {
  double value = -INFINITY;
  double param = 0;
  void add(double value, double param = NAN) {
    if (value <= this->value) return;
    this->value = value;
    this->param = param;
  }
};

struct Testcase;
vector<Testcase *> testcases;

struct Testcase {
  virtual ~Testcase() {}
  Sequence inputs;
  ParamVec ps;
  Sequence outputs;
  Sequence targets;
  virtual string name() { return typeid(*this).name(); }
  // Store random initial test values appropriate for
  // the test case into inputs, ps, and targets
  virtual void init() {
    // reasonable defaults
    randseq(inputs, 1, 7, 4);
    randseq(targets, 1, 3, 4);
    randparams(ps, {{3, 7}});
  }
  // Perform forward and backward steps using inputs,
  // outputs, and ps.
  virtual void forward() = 0;
  virtual void backward() = 0;
};

void test_net(Testcase &tc) {
  int verbose = getienv("verbose", 0);
  print("testing", tc.name());

  tc.init();
  // make backups for computing derivatives
  Sequence inputs = tc.inputs;
  Sequence targets = tc.targets;
  ParamVec ps = tc.ps;

  Maximizer maxinerr;
  int N = inputs.size();
  int ninput = inputs.rows();
  int bs = inputs.cols();
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < ninput; i++) {
      for (int b = 0; b < bs; b++) {
        Minimizer minerr;
        for (float h = 1e-6; h < 1.0; h *= 10) {
          tc.inputs = inputs;
          tc.outputs.like(targets);
          tc.forward();
          double out = mse(tc.outputs, targets);
          tc.inputs.zeroGrad();
          for (Params &p : tc.ps) p.zeroGrad();
          tc.backward();
          double a_deriv = tc.inputs[t].d(i, b);
          tc.inputs[t].v(i, b) += h;
          tc.forward();
          double out1 = mse(tc.outputs, targets);
          double num_deriv = (out1 - out) / h;
          double error = fabs(1.0 - num_deriv / a_deriv / -2.0);
          if (verbose > 1)
            print(t, i, b, ":", error, h, "num:", num_deriv, "analytic:",
                  a_deriv, "out:", out1, out);
          minerr.add(error, h);
        }
        if (verbose) print("deltas", t, i, b, minerr.value, minerr.param);
        assert(minerr.value < 0.1);
        maxinerr.add(minerr.value);
      }
    }
  }

  Maximizer maxparamerr;
  for (int k = 0; k < ps.size(); k++) {
    int n = ps[k].rows();
    int m = ps[k].cols();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        Minimizer minerr;
        for (float h = 1e-6; h < 1.0; h *= 10) {
          tc.ps = ps;
          tc.inputs = inputs;
          tc.outputs.like(targets);
          tc.forward();
          double out = mse(tc.outputs, targets);
          tc.inputs.zeroGrad();
          for (Params &p : tc.ps) p.zeroGrad();
          tc.backward();
          double a_deriv = tc.ps[k].d(i, j);
          tc.ps[k].v(i, j) += h;
          tc.forward();
          double out1 = mse(tc.outputs, targets);
          double num_deriv = (out1 - out) / h;
          double error = fabs(1.0 - num_deriv / a_deriv / -2.0);
          if (verbose > 1)
            print(k, i, j, ":", error, h, "/", num_deriv, a_deriv, out1, out);
          minerr.add(error, h);
        }
        maxparamerr.add(minerr.value);
      }
    }
  }

  tc.inputs = inputs;
  tc.ps = ps;
  tc.targets = targets;

  print("OK", maxinerr.value, maxparamerr.value);
}

struct TestFull1Sigmoid : Testcase {
  virtual void init() {
    randseq(inputs, 1, 7, 4);
    randseq(targets, 1, 3, 4);
    randparams(ps, {{3, 8}});
  }
  void forward() { forward_full1(outputs[0], ps[0], inputs[0], SIG); }
  void backward() { backward_full1(outputs[0], ps[0], inputs[0], SIG); }
};
struct TestFull1Tanh : Testcase {
  virtual void init() {
    randseq(inputs, 1, 7, 4);
    randseq(targets, 1, 3, 4);
    randparams(ps, {{3, 8}});
  }
  void forward() { forward_full1(outputs[0], ps[0], inputs[0], TANH); }
  void backward() { backward_full1(outputs[0], ps[0], inputs[0], TANH); }
};
struct TestFull1Logmag : Testcase {
  virtual void init() {
    randseq(inputs, 1, 7, 4);
    randseq(targets, 1, 3, 4);
    randparams(ps, {{3, 8}});
  }
  void forward() { forward_full1(outputs[0], ps[0], inputs[0], LOGMAG); }
  void backward() { backward_full1(outputs[0], ps[0], inputs[0], LOGMAG); }
};
struct TestStack : Testcase {
  virtual void init() {
    randseq(inputs, 2, 7, 4);
    randseq(targets, 1, 14, 4);
    randparams(ps, {});
  }
  void forward() { forward_stack(outputs[0], inputs[0], inputs[1]); }
  void backward() { backward_stack(outputs[0], inputs[0], inputs[1]); }
};
struct TestStackDelay : Testcase {
  virtual void init() {
    randseq(inputs, 2, 7, 4);
    randseq(targets, 1, 14, 4);
    randparams(ps, {});
  }
  void forward() { forward_stack_delay(outputs[0], inputs[0], inputs, 1); }
  void backward() { backward_stack_delay(outputs[0], inputs[0], inputs, 1); }
};
#ifdef DEPRECATED
struct TestFullSigmoid : Testcase {
  void forward() { forward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0]); }
  void backward() {
    backward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0]);
  }
};
struct TestFullTanh : Testcase {
  void forward() { forward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0]); }
  void backward() {
    backward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0]);
  }
};
struct TestStack1Delay : Testcase {
  virtual void init() {
    randseq(inputs, 2, 7, 4);
    randseq(targets, 1, 15, 4);
    randparams(ps, {});
  }
  void forward() { forward_stack1(outputs[0], inputs[0], inputs, 1); }
  void backward() { backward_stack1(outputs[0], inputs[0], inputs, 1); }
};
#endif
struct TestReverse : Testcase {
  virtual void init() {
    randseq(inputs, 5, 7, 4);
    randseq(targets, 5, 7, 4);
    randparams(ps, {});
  }
  void forward() { forward_reverse(outputs, inputs); }
  void backward() { backward_reverse(outputs, inputs); }
};
struct TestBtswitch : Testcase {
  virtual void init() {
    randseq(inputs, 5, 7, 4);
    randseq(targets, 4, 7, 5);
    randparams(ps, {});
  }
  void forward() { forward_btswitch(outputs, inputs); }
  void backward() { backward_btswitch(outputs, inputs); }
};
struct TestBatchstack : Testcase {
  virtual void init() {
    randseq(inputs, 5, 4, 11);
    randseq(targets, 5, 12, 11);
    randparams(ps, {});
  }
  void forward() { forward_batchstack(outputs, inputs, 1, 1); }
  void backward() { backward_batchstack(outputs, inputs, 1, 1); }
};
struct TestStatemem : Testcase {
  virtual void init() {
    randseq(inputs, 4, 7, 4);
    randseq(targets, 1, 7, 4);
    randparams(ps, {});
  }
  void forward() {
    forward_statemem(outputs[0], inputs[0], inputs[1], inputs, 2, inputs[3]);
  }
  void backward() {
    backward_statemem(outputs[0], inputs[0], inputs[1], inputs, 2, inputs[3]);
  }
};
struct TestNonlingate : Testcase {
  virtual void init() {
    randseq(inputs, 2, 7, 4);
    randseq(targets, 1, 7, 4);
    randparams(ps, {});
  }
  void forward() { forward_nonlingate(outputs[0], inputs[0], inputs[1], TANH); }
  void backward() {
    backward_nonlingate(outputs[0], inputs[0], inputs[1], TANH);
  }
};

inline Eigen::array<ptrdiff_t, 1> indexes(int i) {
  return Eigen::array<ptrdiff_t, 1>({i});
}

inline Eigen::array<ptrdiff_t, 2> indexes(int i, int j) {
  return Eigen::array<ptrdiff_t, 2>({i, j});
}

#ifdef DEPRECATED
void test_full() {
  print("comparing full and full1");
  Sequence inputs;
  ParamVec ps;
  Sequence outputs;
  randseq(inputs, 1, 7, 4);
  randparams(ps, {{3, 8}});
  randseq(outputs, 2, 3, 4);
  Batch inputs1;
  inputs1.resize(8, 4);
  inputs1.v().slice(indexes(0, 0), indexes(1, 4)).setConstant(Float(1));
  inputs1.v().slice(indexes(1, 0), indexes(7, 4)) = inputs[0].v();
  forward_full1<SigmoidNonlin>(outputs[0], ps[0], inputs[0]);
  forward_full<SigmoidNonlin>(outputs[1], ps[0], inputs1);
  EigenTensor1 err = (outputs[0].v() - outputs[1].v()).abs().maximum();
  assert(err(0) < 0.001);
  print("OK", err(0));
  backward_full1<SigmoidNonlin>(outputs[0], ps[0], inputs[0]);
  backward_full<SigmoidNonlin>(outputs[1], ps[0], inputs1);
  EigenTensor1 derr =
      (inputs[0].d() - inputs1.d().slice(indexes(1, 0), indexes(7, 4)))
          .abs()
          .maximum();
  // assert(derr(0) < 0.001);
  print("OK", derr(0));
}
#endif

int main(int argc, char **argv) {
  TRY {
    test_net(*new TestBatchstack);
    test_net(*new TestFull1Sigmoid);
    test_net(*new TestFull1Tanh);
    test_net(*new TestFull1Logmag);
    test_net(*new TestStack);
    test_net(*new TestStackDelay);
    test_net(*new TestReverse);
    test_net(*new TestBtswitch);
    test_net(*new TestStatemem);
    test_net(*new TestNonlingate);
#ifdef DEPRECATED
    test_net(*new TestFullSigmoid);
    test_net(*new TestFullTanh);
    test_net(*new TestStack1Delay);
    test_full();
#endif
  }
  CATCH(const char *message) { print("ERROR", message); }
}
