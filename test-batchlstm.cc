#include <assert.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>     // unlink
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
int nfeatures = getienv("nfeatures", 1);
int trainbatch = getienv("trainbatch", 1);
int testbatch = getienv("testbatch", 1);
int seqlength = getienv("seqlength", 20);
double lrate = getdenv("lrate", 1e-4);

void gentest(Sequence &xs, Sequence &ys, int batchsize = 1) {
  int N = seqlength;
  int d = nfeatures;
  xs.resize(N, d, batchsize);
  xs.zero();
  ys.resize(N, 2, batchsize);
  ys.zero();
  for (int b = 0; b < batchsize; b++) {
    ys[0].v(0, b) = 1;
    for (int t = 0; t < N; t++) {
      int out = (drand48() < 0.3);
      for (int i = 0; i < d; i++) xs[t].v(i, b) = out;
      if (t < N - 1) ys[t + 1].v(out, b) = 1.0;
    }
  }
}

Float maxerr(Sequence &xs, Sequence &ys) {
  Float threshold = getdenv("threshold", 0.1);
  Float merr = 0.0;
  for (int t = 0; t < xs.size(); t++) {
    for (int i = 0; i < xs.rows(); i++) {
      for (int j = 0; j < ys.cols(); j++) {
        Float err = fabs(xs[t].v(i, j) - ys[t].v(i, j));
        if (err > threshold) {
          print("t", t, "i", i, "b", j, "err", err, "xs", xs[t].v(i, j), "ys",
                ys[t].v(i, j));
          assert(err <= threshold);
        }
        merr = fmax(err, merr);
      }
    }
  }
  return merr;
}

void printseq(Sequence &s) {
  for (int i = 0; i < s.rows(); i++) {
    for (int t = 0; t < s.size(); t++) {
      for (int b = 0; b < s.cols(); b++) {
        cerr << std::setw(3) << int(99.999 * s[t].v(i, b));
      }
      cerr << "|";
    }
    cerr << endl;
  }
}

double test_net(Network net) {
  Float merr = 0.0;
  for (int i = 0; i < ntest; i++) {
    Sequence xs, ys;
    gentest(xs, ys, testbatch);
    set_inputs(net, xs);
    net->forward();
    if (getienv("verbose", 0)) {
      print("xs");
      printseq(xs);
      print("ys");
      printseq(ys);
      print("outputs");
      printseq(net->outputs);
      check_normalized(net->outputs);
    }
    Float err = maxerr(net->outputs, ys);
    if (err > merr) merr = err;
  }
  return merr;
}

int main(int argc, char **argv) {
  Network net;
  int gpu = getienv("gpu", -1);
  net = make_net(
      "lstm1",
      {{"ninput", nfeatures}, {"nhidden", 4}, {"noutput", 2}, {"gpu", gpu}});
  net->setLearningRate(lrate, 0.9);
  save_net("__test0__.clstm", net);
  unlink("__test0__.clstm");
  print("training 1:4:2 network to learn delay");
  for (int i = 0; i < ntrain / trainbatch; i++) {
    Sequence xs, ys;
    gentest(xs, ys, trainbatch);
    set_inputs(net, xs);
    net->forward();
    check_normalized(net->outputs);
    set_targets(net, ys);
    net->backward();
    sgd_update(net);
  }
  network_detail(net);
  double merr0 = test_net(net);
  if (merr0 > 0.1) {
    print("FAILED (pre-save)", merr0);
    exit(1);
  } else {
    print("OK (pre-save)", merr0);
  }
  print("saving");
  save_net("__test__.clstm", net);
  net.reset();
  print("loading");
  net = load_net("__test__.clstm");
  double merr = test_net(net);
  unlink("__test__.clstm");
  if (merr > 0.1) {
    print("FAILED", merr);
    exit(1);
  } else {
    print("OK", merr);
  }
}
