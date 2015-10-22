// -*- C++ -*-

// A basic LSTM implementation in C++. All you should need is clstm.cc and
// clstm.h. Library dependencies are limited to a small subset of STL and
// Eigen/Dense

#ifndef ocropus_lstm_
#define ocropus_lstm_

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <memory>
#include <map>
#include <random>
#include "clstm_compute.h"
#include <initializer_list>
#include "enroll.h"

namespace ocropus {
using std::string;
using std::vector;
using std::map;
using std::shared_ptr;
using std::unique_ptr;
using std::function;

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
      THROW("missing parameter");
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

  // Clip and reset gradients in this node.
  void gradientClipParameters(Float value);
  void gradientClipStates(Float value);
  void zeroGradsStates();
  void zeroGradsParameters();

  // Learning rate and momentum used for training.
  int nseq = 0;
  int nsteps = 0;
  Float effective_lr();
  virtual void setLearningRate(Float lr, Float momentum);
  virtual void update();

  // Misc parameters for construction, saving.
  Assoc attr;

  // Networks have input and output "ports" for sequences
  // and derivatives. These are propagated in forward()
  // and backward() methods.
  Sequence inputs;
  Sequence outputs;
  virtual int ninput() { return -999999; }
  virtual int noutput() { return -999999; }

  // Main methods for forward and backward propagation
  // of activations.
  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual void initialize() {}

  // Data for encoding/decoding input/output strings.
  Codec codec, icodec;

  // Loading and saving.
  void save(const char *fname);
  void load(const char *fname);
  virtual void postLoad() {}
};

typedef function<void(const string &, Params *)> ParamsFun;
typedef function<void(const string &, Sequence *)> StateFun;
typedef function<void(const string &, INetwork *)> NetworkFun;
void walk_params(Network net, ParamsFun f, const string &prefix = "");
void walk_states(Network net, StateFun f, const string &prefix = "");
void walk_networks(Network net, NetworkFun f, const string &prefix = "");
void network_info(Network net, string prefix = "");

// setting inputs and outputs
void set_inputs(INetwork *net, Sequence &inputs);
void set_targets(INetwork *net, Sequence &targets);
void set_targets_accelerated(INetwork *net, Sequence &targets);
void set_classes(INetwork *net, Classes &classes);
void set_classes(INetwork *net, BatchClasses &classes);

// instantiating layers and networks

typedef std::function<INetwork *(void)> ILayerFactory;
extern map<string, ILayerFactory> layer_factories;
Network make_layer(const string &kind);

typedef std::vector<Network> Networks;
Network layer(const string &kind, int ninput, int noutput, const Assoc &args,
              const Networks &subs);

typedef std::function<Network(const Assoc &)> INetworkFactory;
extern map<string, INetworkFactory> network_factories;
Network make_net(const string &kind, const Assoc &params);
Network make_net_init(const string &kind, const std::string &params);

// new, proto-based I/O
Network proto_clone_net(INetwork *net);
void debug_as_proto(INetwork *net, bool do_weights = false);
void write_as_proto(std::ostream &output, INetwork *net);
void save_as_proto(const string &fname, INetwork *net);
Network load_as_proto(const string &fname);

void save_net(const string &file, Network net);
Network load_net(const string &file);

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

// setting inputs and outputs
void set_inputs(Network net, Tensor<float,2> &inputs);
void set_targets(Network net, Tensor<float,2> &targets);
void set_targets_accelerated(Network net, Tensor<float,2> &targets);
void set_classes(Network net, Tensor<int,1> &targets);

// single sequence training functions
void mktargets(Tensor<float,2> &seq, Tensor<int,1> &targets, int ndim);
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
