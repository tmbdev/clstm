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
#include <Eigen/Dense>
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

void throwf(const char *format, ...);
extern char exception_message[256];

struct String : public std::string {
  String() {}
  String(const char *s) : std::string(s) {}
  String(const std::string &s) : std::string(s) {}
  String(int x) : std::string(std::to_string(x)) {}
  String(double x) : std::string(std::to_string(x)) {}
  double operator+() { return atof(this->c_str()); }
  operator double() { return atof(this->c_str()); }
  void operator=(const string &value) {
    this->string::operator=(value);
  }
  void operator=(const char *value) {
    this->string::operator=(value);
  }
  void operator=(double value) { *this = std::to_string(value); }
};

struct Assoc : std::map<std::string, String> {
  using std::map<std::string, String>::map;
  Assoc() {}
  Assoc(const string &s);
  String get(string key) const {
    auto it = this->find(key);
    if (it == this->end()) {
      throwf("missing parameter: %s", key.c_str());
    }
    return it->second;
  }
  String get(string key, String dflt) const {
    auto it = this->find(key);
    if (it == this->end()) return dflt;
    return it->second;
  }
  void set(string key, String value) {
    this->operator[](key) = value;
  }
};

typedef Assoc Attributes;

struct ITrainable {
  virtual ~ITrainable() {}

  // This string is used to identify the type for
  // saving/loading. It is set from the registry
  // when the object is allocated through the registry
  // (that assures that the string reflects the name
  // of the type in the registry).
  string kind = "";

  vector<pair<Sequence*,string>> states;
  vector<pair<Params*,string>> parameters;
  void enroll(Sequence &s, const char *name) {
    states.push_back(make_pair(&s,name));
  }
  void enroll(Params &p, const char *name) {
    parameters.push_back(make_pair(&p, name));
  }
  template <class T, typename... Args>
  inline void enroll(T arg, Args... args) {
    enroll(arg);
    enroll(args...);
  }
  virtual void update() {
    Float lr = effective_lr();
    for(auto it : parameters)
      it.first->update(lr, momentum);
  }

  // Learning rate and momentum used for training.
  Float learning_rate = 1e-4;
  Float momentum = 0.9;
  enum Normalization : int {
    NORM_NONE,
    NORM_LEN,
    NORM_BATCH,
    NORM_DFLT = NORM_NONE,
  } normalization = NORM_DFLT;
  int nseq = 0;
  int nsteps = 0;
  Float effective_lr() {
    Float lr = learning_rate;
    if (normalization == NORM_BATCH)
      lr /= fmax(1.0,nseq);
    else if (normalization == NORM_LEN)
      lr /= fmax(1.0,nsteps);
    else if (normalization == NORM_NONE) /* do nothing */
      ;
    else
      THROW("unknown normalization");
    nseq = 0;
    nsteps = 0;
    return lr;
  }
  // The attributes array contains parameters for constructing the
  // network, as well as information necessary for loading and saving
  // networks.
  Attributes attr;

  // Learning rates
  virtual void setLearningRate(Float lr, Float momentum) = 0;

  // Main methods for forward and backward propagation
  // of activations.
  virtual void forward() = 0;
  virtual void backward() = 0;

  virtual int idepth() { return -9999; }
  virtual int odepth() { return -9999; }

  virtual void initialize() {
    // this gets initialization parameters
    // out of the attributes array
  }

  // These are convenience functions for initialization
  virtual void init(int no, int ni) final {
    attr.set("ninput", ni);
    attr.set("noutput", no);
    initialize();
  }
  virtual void init(int no, int nh, int ni) final {
    attr.set("ninput", ni);
    attr.set("nhidden", nh);
    attr.set("noutput", no);
    initialize();
  }
  virtual void init(int no, int nh2, int nh, int ni) final {
    attr.set("ninput", ni);
    attr.set("nhidden", nh);
    attr.set("nhidden2", nh2);
    attr.set("noutput", no);
    initialize();
  }
};

struct INetwork;
typedef shared_ptr<INetwork> Network;

struct INetwork : virtual ITrainable {
  // Networks have input and output "ports" for sequences
  // and derivatives. These are propagated in forward()
  // and backward() methods.
  Sequence inputs;
  Sequence outputs;

  // Some networks have subnetworks. They should be
  // stored in the `sub` vector. That way, functions
  // like `save` can automatically traverse the tree
  // of networks. Together with the `name` field,
  // this forms a hierarchical namespace of networks.
  vector<Network> sub;

  // Data for encoding/decoding input/output strings.
  vector<int> codec;
  vector<int> icodec;
  unique_ptr<map<int, int> > encoder;   // cached
  unique_ptr<map<int, int> > iencoder;  // cached
  void makeEncoders();
  wchar_t decode(int cls);
  wchar_t idecode(int cls);
  std::wstring decode(Classes &cs);
  std::wstring idecode(Classes &cs);
  void encode(Classes &cs, const std::wstring &s);
  void iencode(Classes &cs, const std::wstring &s);

  // Parameters specific to softmax.
  Float softmax_floor = 1e-5;
  bool softmax_accel = false;

  virtual ~INetwork() {}

  std::function<void(INetwork *)> initializer = [](INetwork *) {};
  virtual void initialize() {
    // this gets initialization parameters
    // out of the attributes array
    initializer(this);
  }

  // Expected number of input/output features.
  virtual int ninput() { return -999999; }
  virtual int noutput() { return -999999; }

  // Add a network as a subnetwork.
  virtual void add(Network net) { sub.push_back(net); }

  // Hooks to iterate over the weights and states of this network.
  typedef function<void(const string &, VecMat, VecMat)> WeightFun;
  typedef function<void(const string &, Params *)> ParamsFun;
  typedef function<void(const string &, Sequence *)> StateFun;
  virtual void myweights(const string &prefix, WeightFun f) {}
  virtual void myparams(const string &prefix, ParamsFun f) {}
  virtual void mystates(const string &prefix, StateFun f) {}

  // Hooks executed prior to saving and after loading.
  // Loading iterates over the weights with the `weights`
  // methods and restores only the weights. `postLoad`
  // allows classes to update other internal state that
  // depends on matrix size.
  virtual void preSave() {}
  virtual void postLoad() {}

  // Set the learning rate for this network and all subnetworks.
  virtual void setLearningRate(Float lr, Float momentum) {
    this->learning_rate = lr;
    this->momentum = momentum;
    for (int i = 0; i < sub.size(); i++) sub[i]->setLearningRate(lr, momentum);
  }

  void info(string prefix);
  void weights(const string &prefix, WeightFun f);
  void params(const string &prefix, ParamsFun f);
  void states(const string &prefix, StateFun f);
  void networks(const string &prefix, function<void(string, INetwork *)>);
  Sequence *getState(string name);
  // special method for LSTM and similar networks, returning the
  // primary internal state sequence
  Sequence *getState() {
    throwf("unimplemented");
    return 0;
  };
  void save(const char *fname);
  void load(const char *fname);
};

// standard layer types
INetwork *make_SigmoidLayer();
INetwork *make_SoftmaxLayer();
INetwork *make_ReluLayer();
INetwork *make_Stacked();
INetwork *make_Reversed();
INetwork *make_Parallel();
INetwork *make_LSTM();
INetwork *make_NPLSTM();
INetwork *make_BidiLayer();

// setting inputs and outputs
void set_inputs(INetwork *net, Sequence &inputs);
void set_targets(INetwork *net, Sequence &targets);
void set_targets_accelerated(INetwork *net, Sequence &targets);
void set_classes(INetwork *net, Classes &classes);
void set_classes(INetwork *net, BatchClasses &classes);

// single sequence training functions
void train(INetwork *net, Sequence &xs, Sequence &targets);
void ctrain(INetwork *net, Sequence &xs, Classes &cs);
void ctrain_accelerated(INetwork *net, Sequence &xs, Classes &cs,
                        Float lo = 1e-5);
void cpred(INetwork *net, Classes &preds, Sequence &xs);
void mktargets(Sequence &seq, Classes &targets, int ndim);

// batch training functions
void ctrain(INetwork *net, Sequence &xs, BatchClasses &cs);
void ctrain_accelerated(INetwork *net, Sequence &xs, BatchClasses &cs,
                        Float lo = 1e-5);
void cpred(INetwork *net, BatchClasses &preds, Sequence &xs);
void mktargets(Sequence &seq, BatchClasses &targets, int ndim);

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
void forward_algorithm(Mat &lr, Mat &lmatch, double skip = -5.0);
void forwardbackward(Mat &both, Mat &lmatch);
void ctc_align_targets(Mat &posteriors, Mat &outputs, Mat &targets);
void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Sequence &targets);
void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Classes &targets);
void trivial_decode(Classes &cs, Sequence &outputs, int batch = 0);
void trivial_decode(Classes &cs, Sequence &outputs, int batch,
                    vector<int> *locs);

// DEPRECATED

extern Mat debugmat;

// loading and saving networks (using HDF5)
void load_attributes(map<string, string> &attrs, const string &file);
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
