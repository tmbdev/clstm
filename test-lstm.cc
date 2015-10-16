#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include "extras.h"

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

int ntrain = getienv("ntrain", 100000);
int ntest = getienv("ntest", 1000);

void gentest(Sequence &xs, Sequence &ys) {
  int N = 20;
  xs.resize(N, 1, 1);
  xs.zero();
  ys.resize(N, 2, 1);
  ys.zero();
  ys[0].v(0, 0) = 1;
  for (int t = 0; t < N; t++) {
    int out = (drand48() < 0.3);
    xs[t].v(0, 0) = out;
    if (t < N - 1) ys[t + 1].v(out, 0) = 1.0;
  }
}

Float maxerr(Sequence &xs, Sequence &ys) {
  Float merr = 0.0;
  for (int t = 0; t < xs.size(); t++) {
    for (int i = 0; i < xs.rows(); i++) {
      for (int j = 0; j < ys.cols(); j++) {
        Float err = fabs(xs[t].v(i, j) - ys[t].v(i, j));
        merr = fmax(err, merr);
      }
    }
  }
  return merr;
}

double test_net(Network net) {
  Float merr = 0.0;
  for (int i = 0; i < ntest; i++) {
    Sequence xs, ys;
    gentest(xs, ys);
    set_inputs(net.get(), xs);
    net->forward();
    if (getienv("verbose", 0)) {
      for (int t = 0; t < xs.size(); t++) cout << xs[t].v(0, 0);
      cout << endl;
      for (int t = 0; t < net->outputs.size(); t++)
        cout << int(0.5 + net->outputs[t].v(1, 0));
      cout << endl;
      cout << endl;
    }
    Float err = maxerr(net->outputs, ys);
    assert(err < 0.1);
    if (err > merr) merr = err;
  }
  return merr;
}

int main(int argc, char **argv) {
  Network net = make_net_init("lstm1", "ninput=1:nhidden=4:noutput=2");
  net->setLearningRate(1e-4, 0.9);
  network_info(net);
  save_net("__test0__.clstm", net);
  print("training 1:4:2 network to learn delay");
  for (int i = 0; i < ntrain; i++) {
    Sequence xs, ys;
    gentest(xs, ys);
    set_inputs(net.get(), xs);
    net->forward();
    set_targets(net.get(), ys);
    net->backward();
    net->update();
  }
  test_net(net);
  print("saving");
  save_net("__test__.clstm", net);
  net.reset();
  print("loading");
  net = load_net("__test__.clstm");
  double merr = test_net(net);
  print("OK", merr);
}
