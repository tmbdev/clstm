// Test case for copying parameters and states in/out of the network.

#include <assert.h>
#include <math.h>
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
    set_inputs(net, xs);
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
    if (err > merr) merr = err;
  }
  return merr;
}

#define die() \
  (cerr << "FATAL " << __FILE__ << " " << __LINE__ << "\n", abort(), true)

int main(int argc, char **argv) {
  float learning_rate = 0.01;
  auto factory = [&] {
    Network net = make_net(
        "lstm1", {{"ninput", 1}, {"nhidden", 4}, {"noutput", 2}, {"gpu", -1}});
    net->setLearningRate(learning_rate, 0.0);
    return net;
  };
  Network net = factory();
  print("training 1:4:2 network to learn delay");
  Eigen::Tensor<float, 1> states, weights, derivs;
  for (int i = 0; i < ntrain; i++) {
    Sequence xs, ys;
    gentest(xs, ys);
    set_inputs(net, xs);
    net->forward();
    clear_derivs(net);
    clear_state_derivs(net);

    int nstates = n_states(net);
    int nweights = n_params(net);
    states.resize(nstates);
    weights.resize(nweights);
    derivs.resize(nweights);
    get_states(net, states.data(), nstates);
    get_params(net, weights.data(), nweights);

#ifdef DIRECT
    set_targets(net, ys);
    net->backward();
    if (i == 0) {
      cerr << "DIRECT:\n";
      network_detail(net);
    }
    sgd_update(net);
#endif

    net = factory();
    set_states(net, states.data(), nstates);
    set_params(net, weights.data(), nweights);
    clear_derivs(net);
    clear_state_derivs(net);
    set_targets(net, ys);
    net->backward();
    if (i == 0) {
      cerr << "COPIED:\n";
      network_detail(net);
    }

    // perform stochastic gradient descent on
    // the externalized weights instead of sgd_update(net)
    get_derivs(net, derivs.data(), nweights);
    weights = weights + derivs * Float(0.01);
    set_params(net, weights.data(), nweights);
  }
  // network_detail(net);
  double merr0 = test_net(net);
  if (merr0 > 0.1) {
    print("FAILED (pre-save)", merr0);
    exit(1);
  } else {
    print("OK (pre-save)", merr0);
  }
}
