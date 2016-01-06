#include <assert.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "clstm.h"
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
void randparams(vector<Params> &a) {
  int N = a.size();
  for (int t = 0; t < N; t++) {
    int n = a[t].rows();
    int m = a[t].cols();
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a[t].v(i, j) = randu();
  }
}

double err(Sequence &a, Sequence &b) {
  assert(a.size() == b.size());
  assert(a.rows() == b.rows());
  assert(a.cols() == b.cols());
  int N = a.size(), n = a.rows(), m = a.cols();
  double total = 0.0;
  for (int t = 0; t < N; t++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) total += sqr(a[t].v(i, j) - b[t].v(i, j));
  return total;
}

void zero_grad(Network net) {
  walk_params(net, [](const string &s, Params *p) { p->zeroGrad(); });
}
void get_params(vector<Params> &params, Network net) {
  params.clear();
  walk_params(
      net, [&params](const string &s, Params *p) { params.emplace_back(*p); });
}

void set_params(Network net, vector<Params> &params) {
  int index = 0;
  walk_params(net, [&index, &params](const string &s, Params *p) {
    *p = params[index++];
  });
  assert(index == params.size());
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

void test_net(Network net, string id = "", int N = 4, int bs = 1) {
  if (id == "") id = net->kind;
  print("testing", id);
  int ninput = net->ninput();
  int noutput = net->noutput();
  ;
  bool verbose = getienv("verbose", 0);
  vector<Params> params, params1;
  get_params(params, net);
  randparams(params);
  set_params(net, params);
  Sequence xs, ys;
  randseq(xs, N, ninput, bs);
  randseq(ys, N, noutput, bs);

  Maximizer maxinerr;
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < ninput; i++) {
      for (int b = 0; b < bs; b++) {
        Minimizer minerr;
        for (float h = 1e-6; h < 1.0; h *= 10) {
          set_inputs(net, xs);
          net->forward();
          double out1 = err(net->outputs, ys);
          net->inputs[t].v(i, b) += h;
          net->forward();
          double out2 = err(net->outputs, ys);
          double num_deriv = (out2 - out1) / h;

          set_inputs(net, xs);
          net->forward();
          set_targets(net, ys);
          net->backward();
          double a_deriv = net->inputs[t].d(i, b);
          double error = fabs(1.0 - num_deriv / a_deriv / -2.0);
          minerr.add(error, h);
        }
        if (verbose) print("deltas", t, i, b, minerr.value, minerr.param);
        assert(minerr.value < 0.1);
        maxinerr.add(minerr.value);
      }
    }
  }

  set_inputs(net, xs);
  net->forward();
  double out = err(net->outputs, ys);
  set_targets(net, ys);
  zero_grad(net);
  net->backward();
  get_params(params, net);

  Maximizer maxparamerr;
  for (int k = 0; k < params.size(); k++) {
    Params &p = params[k];
    int n = p.rows(), m = p.cols();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        Minimizer minerr;
        for (float h = 1e-6; h < 1.0; h *= 10) {
          params1 = params;
          params1[k].v(i, j) += h;
          set_params(net, params1);
          net->forward();
          double out1 = err(net->outputs, ys);
          double num_deriv = (out1 - out) / h;
          double a_deriv = params[k].d(i, j);
          double error = fabs(1.0 - num_deriv / a_deriv / -2.0);
          minerr.add(error, h);
        }
        if (verbose) print("params", k, i, j, minerr.value, minerr.param);
        assert(minerr.value < 0.1);
        maxparamerr.add(minerr.value);
      }
    }
  }
  print("OK", maxinerr.value, maxparamerr.value);
}

int main(int argc, char **argv) {
  TRY {
    test_net(
        make_net("perplstm", {{"ninput", 3}, {"nhidden", 4}, {"noutput", 5}}),
        "perplstm", 11, 13);
    test_net(make_net("twod", {{"ninput", 3},
                               {"nhidden", 4},
                               {"noutput", 5},
                               {"output_type", "SigmoidLayer"}}),
             "twod", 11, 13);
    test_net(layer("LinearLayer", 7, 3, {}, {}));
    test_net(layer("SigmoidLayer", 7, 3, {}, {}));
    test_net(layer("TanhLayer", 7, 3, {}, {}));
    test_net(layer("NPLSTM", 7, 3, {}, {}));
    test_net(
        layer("Reversed", 7, 3, {}, {layer("SigmoidLayer", 7, 3, {}, {})}));
    test_net(layer("Parallel", 7, 3, {}, {layer("SigmoidLayer", 7, 3, {}, {}),
                                          layer("LinearLayer", 7, 3, {}, {})}),
             "parallel(sigmoid,linear)");
    test_net(make_net("bidi", {{"ninput", 7},
                               {"noutput", 3},
                               {"nhidden", 5},
                               {"output_type", "SigmoidLayer"}}),
             "bidi");
    test_net(layer("Stacked", 3, 3, {}, {layer("Btswitch", 3, 3, {}, {}),
                                         layer("Btswitch", 3, 3, {}, {})}),
             "btswitch");
    test_net(layer("Batchstack", 3, 9, {}, {}), "Batchstack", 4, 5);
    // not testing: SoftmaxLayer and ReluLayer
  }
  CATCH(const char *message) { print("ERROR", message); }
}
