#include "clstm.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "extras.h"
#include "utils.h"

namespace ocropus {
map<string, INetworkFactory> network_factories;

string get(const Assoc &params, const string &key, const string &dflt) {
  auto it = params.find(key);
  if (it == params.end()) return dflt;
  return it->second;
}

// A 1D unidirectional LSTM with Softmax/Sigmoid output layer.

Network make_lstm1(const Assoc &params) {
  int ninput = params.get("ninput");
  int nhidden = params.get("nhidden");
  int noutput = params.get("noutput");
  string lstm_type = get(params, "lstm_type", "NPLSTM");
  string output_type = get(params, "output_type",
                           noutput == 1 ? "SigmoidLayer" : "SoftmaxLayer");
  return layer("Stacked", ninput, noutput, {},
               {layer(lstm_type, ninput, nhidden, params, {}),
                layer(output_type, nhidden, noutput, params, {})});
}

// A 1D unidirectional reversed LSTM with Softmax/Sigmoid output layer.

Network make_revlstm1(const Assoc &params) {
  int ninput = params.get("ninput");
  int nhidden = params.get("nhidden");
  int noutput = params.get("noutput");
  string lstm_type = get(params, "lstm_type", "NPLSTM");
  string output_type = get(params, "output_type",
                           noutput == 1 ? "SigmoidLayer" : "SoftmaxLayer");
  return layer("Stacked", ninput, noutput, {},
               {layer("Reversed", ninput, nhidden, {},
                      {layer(lstm_type, ninput, nhidden, params, {})}),
                layer(output_type, nhidden, noutput, params, {})});
}

// A 1D bidirectional LSTM with Softmax/Sigmoid output layer.

Network make_bidi(const Assoc &params) {
  int ninput = params.get("ninput");
  int nhidden = params.get("nhidden");
  int noutput = params.get("noutput");
  string lstm_type = get(params, "lstm_type", "NPLSTM");
  string output_type = get(params, "output_type",
                           noutput == 1 ? "SigmoidLayer" : "SoftmaxLayer");
  return layer(
      "Stacked", ninput, noutput, {},
      {layer("Parallel", ninput, 2 * nhidden, {},
             {
                 layer(lstm_type, ninput, nhidden, params, {}),
                 layer("Reversed", ninput, ninput, {},
                       {layer(lstm_type, ninput, nhidden, params, {})}),
             }),
       layer(output_type, 2 * nhidden, noutput, params, {})});
}

// A 1D bidirectional LSTM with Softmax/Sigmoid output layer.

Network make_bidi0(const Assoc &params) {
  int ninput = params.get("ninput");
  int noutput = params.get("noutput");
  string lstm_type = get(params, "lstm_type", "NPLSTM");
  return layer("Parallel", ninput, 2 * noutput, {},
               {
                   layer(lstm_type, ninput, noutput, params, {}),
                   layer("Reversed", ninput, ninput, {},
                         {layer(lstm_type, ninput, noutput, params, {})}),
               });
}

// Two stacked 1D bidirectional LSTM with Softmax/Sigmoid output layer.

Network make_bidi2(const Assoc &params) {
  int ninput = params.get("ninput");
  int nhidden = params.get("nhidden");
  int nhidden2 = params.get("nhidden2");
  int noutput = params.get("noutput");
  string lstm_type = get(params, "lstm_type", "NPLSTM");
  string output_type = get(params, "output_type",
                           noutput == 1 ? "SigmoidLayer" : "SoftmaxLayer");
  return layer(
      "Stacked", ninput, noutput, {},
      {layer("Parallel", ninput, 2 * nhidden, {},
             {
                 layer(lstm_type, ninput, nhidden, params, {}),
                 layer("Reversed", ninput, ninput, {},
                       {layer(lstm_type, ninput, nhidden, params, {})}),
             }),
       layer("Parallel", 2 * nhidden, 2 * nhidden2, {},
             {
                 layer(lstm_type, 2 * nhidden, nhidden2, params, {}),
                 layer("Reversed", 2 * nhidden, 2 * nhidden, {},
                       {layer(lstm_type, 2 * nhidden, nhidden2, params, {})}),
             }),
       layer(output_type, 2 * nhidden2, noutput, params, {})});
}

Network make_perplstm(const Assoc &params) {
  int ninput = params.get("ninput");
  int nhidden = params.get("nhidden");
  int noutput = params.get("noutput");
  string output_type = get(params, "output_type", "SigmoidLayer");
  Network vertical = make_bidi({{"ninput", ninput},
                                {"nhidden", nhidden},
                                {"noutput", noutput},
                                {"output_type", output_type}});
  return layer("Stacked", ninput, noutput, {},
               {
                   // layer("Btswitch", nhidden2, nhidden2, {}, {}),
                   vertical,
                   // layer("Btswitch", noutput, noutput, {}, {})
               });
}

// Two dimensional LSTM

Network make_twod(const Assoc &params) {
  int ninput = params.get("ninput");
  int nhidden = params.get("nhidden");
  int nhidden2 = params.get("nhidden2", nhidden);
  int nhidden3 = params.get("nhidden3", nhidden2);
  int noutput = params.get("noutput");
  string output_type = get(params, "output_type",
                           noutput == 1 ? "SigmoidLayer" : "SoftmaxLayer");
  Network horizontal = make_bidi({{"ninput", ninput},
                                  {"nhidden", nhidden},
                                  {"noutput", nhidden2},
                                  {"output_type", "SigmoidLayer"}});
  Network vertical = make_bidi({{"ninput", nhidden2},
                                {"nhidden", nhidden3},
                                {"noutput", noutput},
                                {"output_type", output_type}});
  return layer("Stacked", ninput, noutput, {},
               {horizontal, layer("Btswitch", nhidden2, nhidden2, {}, {}),
                vertical, layer("Btswitch", noutput, noutput, {}, {})});
}

void init_clstm_prefab() {
  network_factories["lstm1"] = make_lstm1;
  network_factories["revlstm1"] = make_revlstm1;
  network_factories["bidi"] = make_bidi;
  network_factories["bidi0"] = make_bidi0;
  network_factories["bidi2"] = make_bidi2;
  network_factories["twod"] = make_twod;
  network_factories["perplstm"] = make_perplstm;
}

static int init_ = (init_clstm_prefab(), 0);

Network make_net(const string &kind, const Assoc &args) {
  Network result;
  if (network_factories.find(kind) != network_factories.end()) {
    result = network_factories[kind](args);
  } else {
    result = layer(kind, args.get("ninput"), args.get("noutput"), args, {});
  }
  if (!result) throwf("no such network or layer: %s", kind.c_str());
  result->attr.set("kind", kind);
  return result;
}

// Make a network, using parameter specifications of the
// form "ninput=28:noutput=10:nhidden=50:output_type=SoftmaxLayer"

Network make_net_init(const string &kind, const string &params) {
  using std::cerr;
  using std::endl;
  Assoc args(params);
  if (getienv("verbose_params", 0)) {
    for (auto it : args) {
      cerr << it.first << ": " << it.second << endl;
    }
  }
  return make_net(kind, args);
}
}
