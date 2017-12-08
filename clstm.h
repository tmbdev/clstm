// -*- C++ -*-

// A basic LSTM implementation in C++. All you should need is clstm.cc and
// clstm.h. Library dependencies are limited to a small subset of STL and
// Eigen/Dense

#ifndef ocropus_lstm_
#define ocropus_lstm_

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "batches.h"
#include "clstm_compute.h"
#include "enroll.h"

namespace ocropus {
using std::string;
using std::wstring;
using std::vector;
using std::map;
using std::shared_ptr;
using std::unique_ptr;
using std::function;

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

// A string that automatically converts to numbers when needed;
// used for holding parameter values.
class String : public std::string {
 public:
  String() {}
  String(const char *s) : std::string(s) {}
  String(const std::string &s) : std::string(s) {}
  String(int x) : std::string(std::to_string(x)) {}
  String(double x) : std::string(std::to_string(x)) {}
  double operator+() { return atof(this->c_str()); }
  operator double() { return atof(this->c_str()); }
  void operator=(const string &value) { this->string::operator=(value); }
  void operator=(const char *value) { this->string::operator=(value); }
  void operator=(double value) { *this = std::to_string(value); }
};

// A key-value store with defaults.
class Assoc : public std::map<std::string, String> {
 public:
  using std::map<std::string, String>::map;
  Assoc() {}
  Assoc(const string &s);
  Assoc *super = nullptr;
  bool contains(const string &key, bool parent = true) const {
    auto it = this->find(key);
    if (it != this->end()) return true;
    if (parent) return super->contains(key, parent);
    return false;
  }
  String get(const string &key) const {
    auto it = this->find(key);
    if (it == this->end()) {
      if (super) return super->get(key);
      throwf("missing parameter: %s", key.c_str());
    }
    return it->second;
  }
  String get(const string &key, String dflt) const {
    auto it = this->find(key);
    if (it == this->end()) {
      if (super) return super->get(key, dflt);
      return dflt;
    }
    return it->second;
  }
  void set(const string &key, String value) { this->operator[](key) = value; }
};

// A small class for encoding/decoding strings.
class Codec {
 public:
  vector<int> codec;
  unique_ptr<map<int, int> > encoder;
  int size() { return codec.size(); }
  void set(const vector<int> &data);
  wchar_t decode(int cls);
  std::wstring decode(Classes &cs);
  void encode(Classes &cs, const std::wstring &s);
  void build(const vector<string> &fname, const wstring &extra = L"");
};

// The main network interface and a shared_ptr version of it.
class INetwork;
typedef shared_ptr<INetwork> Network;

class INetwork {
 public:
  virtual ~INetwork() {}

  // String that can be used for constructing these objects in `layer`;
  // set when allocated via the registry.
  string kind = "";

  // Networks may have subnetworks, internal states, and parameters.
  vector<Network> sub;
  map<string, Sequence *> states;
  map<string, Params *> parameters;

  // Utility functions for adding subnetworks, etc.
  // (The ENROLL macro makes this easy.)
  virtual void add(Network net) { sub.push_back(net); }
  void enroll(Sequence &s, const char *name) { states[name] = &s; }
  void enroll(Params &p, const char *name) { parameters[name] = &p; }

  // Learning rate and momentum used for training.
  int nseq = 0;
  int nsteps = 0;
  Float effective_lr();
  virtual void setLearningRate(Float lr, Float momentum);

  // Misc parameters for construction, saving.
  Assoc attr;

  // Networks have input and output "ports" for sequences
  // and derivatives. These are propagated in forward()
  // and backward() methods.
  Sequence inputs;
  Sequence outputs;
  virtual int ninput() { return attr.get("ninput"); }
  virtual int noutput() { return attr.get("noutput"); }

  // Main methods for forward and backward propagation
  // of activations.
  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual void initialize() {}

  // Clearing
  void clearStates();
  void clearStateDerivs();
  void clearWeightDerivs();

  // Data for encoding/decoding input/output strings.
  Codec codec, icodec;

  // Loading and saving.
  void save(const char *fname);
  void load(const char *fname);
  virtual void postLoad() {}
};

// The walk_... functions iterate over the internal parameters,
// sequences, or networks of a network. Use as in,
// walk_states(net, [&] (const string &name, Sequence *seq) { ... }));
typedef function<void(const string &, Params *)> ParamsFun;
typedef function<void(const string &, Sequence *)> StateFun;
typedef function<void(const string &, INetwork *)> NetworkFun;

// Walks through all the parameters of the network. Note
// that this gives you both values and derivatives.
void walk_params(Network net, ParamsFun f, const string &prefix = "");

// Walks through any internal state sequences of the network.
// If `io=true`, also walks through the inputs/outputs of all
// layers.
void walk_states(Network net, StateFun f, const string &prefix = "",
                 bool io = false);
// Walks through all the layers (sub-networks) of the network.
void walk_networks(Network net, NetworkFun f, const string &prefix = "");

// output information about the network (for debugging)
void network_info(Network net, string prefix);
void network_detail(Network net, string prefix = "");

// get the number of parameters in a network
int n_params(Network net);

// clear the parameter derivatives in a network
void clear_derivs(Network net);

// given a vector of floating point numbers, alias the internal
// weights to that vector; total must be the size and should be
// equal to the result of n_params; gpu requests GPU storage
// (total and gpu arguments mean the same in the following functions)
void share_params(Network net, Float *params, int total, int gpu = -1);

// set the internal weights of the network from the params argument
void set_params(Network net, const Float *params, int total, int gpu = -1);

// store the internal weights of the network in the params argument
void get_params(Network net, Float *params, int total, int gpu = -1);

// restore the internal weights derivatives from the params argument
void set_derivs(Network net, Float *params, int total, int gpu = -1);

// store the internal weights derivatives in the params argument
void get_derivs(Network net, Float *params, int total, int gpu = -1);

// get the number of internal state variables for the network; this
// changes after every forward propagation pass; note that this includes
// both numerical values and the shapes of internal tensors
int n_states(Network net);

// set the internal state variables of the network from the params argument
void set_states(Network net, const Float *params, int total, int gpu = -1);

// store the internal state variables in the params argument
void get_states(Network net, Float *params, int total, int gpu = -1);

// clear all internal states
void clear_states(Network net);

// invalidate the derivatives in the network
void invalidate_state_derivs(Network net);

// invalidate the derivatives in the network
void clear_state_derivs(Network net);

// set the class targets of the network using hot-1 encoding
void set_classes(Network net, BatchClasses &classes);

// set the class targets of the network using hot-1 encoding, batch size=1
void set_classes(Network net, Classes &classes);

// set the class targets of the network using hot-1 encoding, batch size=1
void set_classes(Network net, Tensor<int, 1> &targets);

// set the inputs of the network from a given Sequence
void set_inputs(Network net, Sequence &inputs);

// set the inputs of the network from a rank 2 tensor (i.e., bath size = 1)
void set_inputs(Network net, TensorMap2 inputs);

// set the targets of the network from a Sequence; this computes deltas
void set_targets(Network net, Sequence &targets);

// set the targets of the network from a rank-2 Tensor (i.e., batch size = 1)
void set_targets(Network net, TensorMap2 targets);

// like set_targets, but using RNNLIB's "accelerated" deltas
void set_targets_accelerated(Network net, Sequence &targets);
void set_targets_accelerated(Network net, TensorMap2 targets);

// update weights inside the network using stochastic gradient descent
void sgd_update(Network net);

// instantiating layers and networks

// a function capable of making a layer
typedef std::function<INetwork *(void)> ILayerFactory;

// a registry for layer factories
extern map<string, ILayerFactory> layer_factories;

// given the name of a layer, produce an instance of its class
Network make_layer(const string &kind);

typedef std::vector<Network> Networks;

// constructs a layer with the given parameters (ninput, noutput, args)
// and adds the sublayer given in the last argument to it; this can be
// used to construct entire network architectures in a single call
Network layer(const string &kind, int ninput, int noutput, const Assoc &args,
              const Networks &subs);

// network_factories is a registry for pre-constructed networks
// (as opposed to layer_factories, which is a registry for individual
// layers); if you have new standard network constructions, pick
// a name and add your network constructor here
typedef std::function<Network(const Assoc &)> INetworkFactory;
extern map<string, INetworkFactory> network_factories;

// construct a network of the given kind with the given parameters
Network make_net(const string &kind, const Assoc &params);

// construct a network of the given kind with the parameters given
// as a single string of the form "name1=value1:name2=value2"
Network make_net_init(const string &kind, const std::string &params);

// write a network to a stream as protocol buffers
bool write_as_proto(std::ostream &output, INetwork *net);

// read a network from an input stream as protocol buffers
Network read_as_proto(std::istream &input);

// convenience functions taking file names instead of streams
bool save_as_proto(const string &fname, INetwork *net);
Network load_as_proto(const string &fname);

// write a network out as protocol buffer text
void debug_as_proto(INetwork *net, bool do_weights = false);

// clone a network via protocol buffers
Network proto_clone_net(Network net);

// convenience functions using Network instead of INetwork *
void save_net(const string &file, Network net);
Network load_net(const string &file);

// functions that return booleans instead of throwing exceptions
bool maybe_save_net(const string &file, Network net);
Network maybe_load_net(const string &file);

// training with CTC
// void forward_algorithm(Mat &lr, Mat &lmatch, double skip = -5.0);
//  void forwardbackward(Mat &both, Mat &lmatch);
// void ctc_align_targets(Mat &posteriors, Mat &outputs, Mat &targets);
void mktargets(Sequence &seq, Classes &transcript, int ndim);
void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Sequence &targets);
void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Classes &targets);
void trivial_decode(Classes &cs, Sequence &outputs, int batch = 0);
void trivial_decode(Classes &cs, Sequence &outputs, int batch,
                    vector<int> *locs);

// single sequence training functions
void mktargets(Tensor2 &seq, Tensor<int, 1> &targets, int ndim);

void share_deltas(vector<Network> &networks);
void average_weights(vector<Network> &networks);
void distribute_weights(vector<Network> &networks, int from = 0);
}

namespace {

template <class A, class B>
double levenshtein(A &a, B &b) {
  using std::vector;
  int n = a.size();
  int m = b.size();
  if (n > m) return levenshtein(b, a);
  vector<double> current(n + 1);
  vector<double> previous(n + 1);
  for (int k = 0; k < current.size(); k++) current[k] = k;
  for (int i = 1; i <= m; i++) {
    previous = current;
    for (int k = 0; k < current.size(); k++) current[k] = 0;
    current[0] = i;
    for (int j = 1; j <= n; j++) {
      double add = previous[j] + 1;
      double del = current[j - 1] + 1;
      double change = previous[j - 1];
      if (a[j - 1] != b[i - 1]) change = change + 1;
      current[j] = fmin(fmin(add, del), change);
    }
  }
  return current[n];
}
}

#endif
