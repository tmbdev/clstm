#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <stdarg.h>

namespace ocropus {
map<string, INetworkFactory> network_factories;

string get(const Assoc &params, const string &key, const string &dflt) {
    auto it = params.find(key);
    if (it == params.end()) return dflt;
    return it->second;
}

// A 1D unidirectional LSTM with Softmax output layer.

Network make_lstm1(const Assoc &params) {
    int ninput = params.at("ninput");
    int nhidden = params.at("nhidden");
    int noutput = params.at("noutput");
    string lstm_type = get(params, "lstm_type", "NPLSTM");
    string output_type = get(params, "output_type", "SoftmaxLayer");
    return layer("Stacked", ninput, noutput, {}, {
                     layer(lstm_type, ninput, nhidden, params, {}),
                     layer(output_type, nhidden, noutput, params, {})
                 });
}

// A 1D unidirectional reversed LSTM with Softmax output layer.

Network make_revlstm1(const Assoc &params) {
    int ninput = params.at("ninput");
    int nhidden = params.at("nhidden");
    int noutput = params.at("noutput");
    string lstm_type = get(params, "lstm_type", "NPLSTM");
    string output_type = get(params, "output_type", "SoftmaxLayer");
    return layer("Stacked", ninput, noutput, {}, {
                     layer("Reversed", ninput, nhidden, {}, {
                               layer(lstm_type, ninput, nhidden, params, {})
                           }),
                     layer(output_type, nhidden, noutput, params, {})
                 });
}

// A 1D bidirectional LSTM with Softmax output layer.

Network make_bidi(const Assoc &params) {
    int ninput = params.at("ninput");
    int nhidden = params.at("nhidden");
    int noutput = params.at("noutput");
    string lstm_type = get(params, "lstm_type", "NPLSTM");
    string output_type = get(params, "output_type", "SoftmaxLayer");
    return layer("Stacked", ninput, noutput, {}, {
                     layer("Parallel", ninput, 2*nhidden, {}, {
                               layer(lstm_type, ninput, nhidden, params, {}),
                               layer("Reversed", ninput, ninput, {}, {
                                         layer(lstm_type, ninput, nhidden, params, {})
                                     }),
                           }),
                     layer(output_type, 2*nhidden, noutput, params, {})
                 });
}

// Two stacked 1D bidirectional LSTM with Softmax output layer.

Network make_bidi2(const Assoc &params) {
    int ninput = params.at("ninput");
    int nhidden = params.at("nhidden");
    int nhidden2 = params.at("nhidden2");
    int noutput = params.at("noutput");
    string lstm_type = get(params, "lstm_type", "NPLSTM");
    string output_type = get(params, "output_type", "SoftmaxLayer");
    return layer("Stacked", ninput, noutput, {}, {
                     layer("Parallel", ninput, 2*nhidden, {}, {
                               layer(lstm_type, ninput, nhidden, params, {}),
                               layer("Reversed", ninput, ninput, {}, {
                                         layer(lstm_type, ninput, nhidden, params, {})
                                     }),
                           }),
                     layer("Parallel", 2*nhidden, 2*nhidden2, {}, {
                               layer(lstm_type, 2*nhidden, nhidden2, params, {}),
                               layer("Reversed", 2*nhidden, 2*nhidden, {}, {
                                         layer(lstm_type, 2*nhidden, nhidden2, params, {})
                                     }),
                           }),
                     layer(output_type, 2*nhidden2, noutput, params, {})
                 });
}

void init_clstm_prefab() {
    network_factories["lstm1"] = make_lstm1;
    network_factories["revlstm1"] = make_revlstm1;
    network_factories["bidi"] = make_bidi;
    network_factories["bidi2"] = make_bidi2;
}

static int init_ = (init_clstm_prefab(), 0);

Network make_net(const string &kind, const Assoc &args) {
    Network result;
    if (network_factories.find(kind) != network_factories.end()) {
        result = network_factories[kind](args);
    } else {
        result = layer(kind, args.at("ninput"), args.at("noutput"), args, {});
    }
    if (!result) throwf("%s: no such network or layer", kind.c_str());
    result->attributes["kind"] = kind;
    return result;
}

// Make a network, using parameter specifications of the
// form "ninput=28:noutput=10:nhidden=50:output_type=SoftmaxLayer"

Network make_net_init(const string &kind, const string &params) {
    using std::cerr;
    using std::endl;
    Assoc args(params);
    for (auto it : args) {
        cerr << it.first << ": " << it.second << endl;
    }
    return make_net(kind, args);
}
}
