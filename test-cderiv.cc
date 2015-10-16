#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include "extras.h"
#include "clstm_compute.h"

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
  a.resize(N, n, m);
  for (int t = 0; t < N; t++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a[t].v(i, j) = randu();
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
  double value = 1e9;
  double param = 0;
  void add(double value, double param = NAN) {
    if (value >= this->value) return;
    this->value = value;
    this->param = param;
  }
};

struct Maximizer {
  double value = -1e9;
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
  Sequence outputs;
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
          outputs = tc.outputs;
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
            print(t, i, b, ":", error, h, "/", num_deriv, a_deriv, out1, out);
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
        }
      }
    }
  }
  print("OK", maxinerr.value, maxparamerr.value);
}

struct TestFullSigmoid : Testcase {
  void forward() { forward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0]); }
  void backward() {
    backward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0], 100.0);
  }
};
struct TestFullTanh : Testcase {
  void forward() { forward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0]); }
  void backward() {
    backward_full<SigmoidNonlin>(outputs[0], ps[0], inputs[0], 100.0);
  }
};
struct TestFull1Sigmoid : Testcase {
  virtual void init() {
    randseq(inputs, 1, 7, 4);
    randseq(targets, 1, 3, 4);
    randparams(ps, {{3, 8}});
  }
  void forward() { forward_full1<SigmoidNonlin>(outputs[0], ps[0], inputs[0]); }
  void backward() {
    backward_full1<SigmoidNonlin>(outputs[0], ps[0], inputs[0], 100.0);
  }
};
struct TestFull1Tanh : Testcase {
  virtual void init() {
    randseq(inputs, 1, 7, 4);
    randseq(targets, 1, 3, 4);
    randparams(ps, {{3, 8}});
  }
  void forward() { forward_full1<SigmoidNonlin>(outputs[0], ps[0], inputs[0]); }
  void backward() {
    backward_full1<SigmoidNonlin>(outputs[0], ps[0], inputs[0], 100.0);
  }
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
struct TestStack1 : Testcase {
  virtual void init() {
    randseq(inputs, 2, 7, 4);
    randseq(targets, 1, 15, 4);
    randparams(ps, {});
  }
  void forward() { forward_stack1(outputs[0], inputs[0], inputs, 1); }
  void backward() { backward_stack1(outputs[0], inputs[0], inputs, 1); }
};

int main(int argc, char **argv) {
  TRY {
    test_net(*new TestFullSigmoid);
    test_net(*new TestFullTanh);
    test_net(*new TestFull1Sigmoid);
    test_net(*new TestFull1Tanh);
    test_net(*new TestStack);
    test_net(*new TestStack1);
  }
  CATCH(const char *message) { print("ERROR", message); }
}
