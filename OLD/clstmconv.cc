#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>

#include "multidim.h"
#include "h5multi.h"
#include "pymulti.h"
#include "extras.h"

using namespace std;
using namespace Eigen;
using namespace ocropus;
using namespace h5multi;
using namespace pymulti;

struct SeqDataset {
  string iname = "inputs";
  string oname = "outputs";
  HDF5 h5;
  int nsamples = -1;
  int nin = -1;
  int nout = -1;

  SeqDataset(const string &h5file) {
    H5::Exception::dontPrint();
    h5.open(h5file);
    mdarray<int> idims, odims;
    h5.shape(idims, iname);
    h5.shape(odims, oname);
    assert(idims(0) == odims(0));
    nsamples = idims(0);
    print("got", nsamples, "training samples");
    mdarray<float> a;
    h5.getdrow(a, 0, iname);
    assert(a.rank() == 2);
    nin = a.dim(1);
    h5.getdrow(a, 0, oname);
    assert(a.rank() == 2);
    nout = a.dim(1);
  }
  void input(mdarray<float> &a, int index) {
    h5.getdrow(a, index, iname);
    assert(a.rank() == 2);
    assert(a.dim(1) == nin);
  }
  void output(mdarray<float> &a, int index) {
    h5.getdrow(a, index, oname);
    assert(a.rank() == 2);
    assert(a.dim(1) == nout);
  }
};

inline void getslice(mdarray<float> &a, Sequence &seq, int i, int b = 0) {
  a.resize(int(seq.size()));
  for (int t = 0; t < seq.size(); t++) a(t) = seq[t](i, b);
}

inline void getslice(mdarray<float> &a, mdarray<float> &seq, int i, int b = 0) {
  if (a.rank() == 2) {
    a.resize(seq.dim(0));
    for (int t = 0; t < seq.size(); t++) a(t) = seq(t, i);
  } else if (a.rank() == 3) {
    a.resize(seq.dim(0));
    for (int t = 0; t < seq.size(); t++) a(t) = seq(t, i, b);
  } else {
    THROW("bad rank");
  }
}

void run_eval(INetwork *net, SeqDataset *dataset, int testmod, int at = -1,
              int ntest = 1000000) {
  mdarray<float> inputs, outputs, targets;
  double total_mse = 0.0;
  double total_mse0 = 0.0;
  double count = 0;
  for (int i = 0; i < dataset->nsamples; i += testmod) {
    if (count >= ntest) break;
    dataset->input(inputs, i);
    dataset->output(targets, i);
    set_inputs(net, inputs);
    net->forward();
    assign(outputs, net->outputs);
    count += 1;
    {
      double err = 0.0, err0 = 0.0;
      for (int i = 0; i < outputs.dim(0); i++) {
        for (int j = 0; j < outputs.dim(1); j++) {
          err += pow(outputs(i, j) - targets(i, j), 2);
          err0 += pow(targets(i, j), 2);
        }
      }
      err /= outputs.dim(0);
      err0 /= outputs.dim(0);
      total_mse += err;
      total_mse0 += err0;
    }
  }
  print("TESTERR", total_mse / count, total_mse / total_mse0, "N", count, "at",
        at);
}

double mse(mdarray<float> &a, mdarray<float> &b) {
  assert(a.size() == b.size());
  double total = 0.0;
  for (int i = 0; i < a.size(); i++) total += pow(a[i] - b[i], 2);
  total /= a.size();
  return total;
}

int main_seq(int argc, char **argv) {
  float lr = getdenv("lrate", 1e-4);
  int display_every = getienv("display_every", 0);
  int report_every = getienv("report_every", 100);
  int ntrain = getienv("ntrain", 10000000);

  int testmod = getienv("testmod", 10);
  int test_every = getienv("test_every", 1000);
  int ntest = getienv("ntest", 1000000);

  PyServer py;
  if (display_every > 0) py.open();

  assert(argc > 1);
  SeqDataset dataset(argv[1]);
  print("ninput", dataset.nin, "noutput", dataset.nout);
  Network net;
  string net_type = getoneof("lstm", "BIDILSTM");
  string lstm_type = getoneof("lstm_type", "LSTM");
  string output_type = getoneof("output_type", "LinearLayer");
  int nhidden = getrenv("nhidden", getrenv("hidden", 100));
  int nhidden2 = getrenv("nhidden2", getrenv("hidden2", -1));
  net = make_net(net_type, {
                            {"ninput", dataset.nin},
                            {"noutput", dataset.nout},
                            {"nhidden", nhidden},
                            {"nhidden2", nhidden2},
                           });

  double lrate = getdenv("lrate", 1e-4);
  net->setLearningRate(lrate, 0.9);

  if (getienv("debug_nets")) {
    net->networks("net", [](string name, INetwork *net) {
      print("net", name, net->learning_rate, net->momentum);
    });
  }
  if (getienv("debug_states")) {
    net->states("net", [](string name, Sequence *seq) {
      print("state", name, seq->size());
    });
  }

  mdarray<float> input, output, target;
  Classes classes;
  int next_test = test_every;
  for (int trial = 0; trial < ntrain; trial++) {
    if (test_every > 0 && trial >= next_test) {
      run_eval(net.get(), &dataset, testmod, trial, ntest);
      next_test += test_every;
      continue;
    }
    int sample;
    do {
      sample = irandom() % dataset.nsamples;
    } while (sample % testmod == 0);
    mdarray<float> input, target;
    dataset.input(input, sample);
    dataset.output(target, sample);
    set_inputs(net.get(), input);
    net->forward();
    assign(output, net->outputs);
    set_targets(net.get(), target);
    net->backward();
    net->update();
    if (report_every > 0 && trial % report_every == 0) {
      print(trial, mse(output, target));
    }
    if (display_every > 0 && trial % display_every == 0) {
      py.eval("clf()");
      py.eval("subplot(121)");
      py.eval("ylim(-.1,1.1)");
      mdarray<float> a;
      getslice(a, input, 0);
      py.plot(a, "color='y',linewidth=5");
      getslice(a, target, 1);
      py.plot(a, "color='b',linewidth=2");
      getslice(a, output, 0);
      py.plot(a, "color='r',ls='--'");
      Sequence *state = 0;
      state = net->getState(".lstm1.0.lstm.state");
      if (state) {
        mdarray<float> b;
        getslice(a, *state, 0);
        getslice(b, *state, 1);
        py.eval("subplot(122)");
        py.plot2(a, b);
      }
      py.eval("ginput(1,0.005)");
    }
  }
  run_eval(net.get(), &dataset, testmod, ntrain);
}

const char *usage = /*program+*/
    "data.h5\n\n"
    "data.h5 is an HDF5 file containing:\n\n"
    "float inputs(N,*): input sequences\n"
    "int inputs_dims(N,2): shape of input sequences\n"
    "float outputs(N,*): output sequences\n"
    "int outputs_dims(N,2): shape of output sequences\n";

int main(int argc, char **argv) {
  if (argc < 2) {
    print(string(argv[0]) + " " + usage);
    exit(1);
  }
  try {
    return main_seq(argc, argv);
  } catch (const char *msg) {
    print("EXCEPTION", msg);
  } catch (...) {
    print("UNKNOWN EXCEPTION");
  }
}
