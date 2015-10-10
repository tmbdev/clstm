#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>
#include <set>
#include <fstream>
#include "pstring.h"

#ifndef MAXEXP
#define MAXEXP 30
#endif

#include <sys/time.h>
inline double now() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

namespace ocropus {
char exception_message[256];

void throwf(const char *format, ...) {
  va_list arglist;
  va_start(arglist, format);
  vsprintf(exception_message, format, arglist);
  va_end(arglist);
  THROW(exception_message);
}

Assoc::Assoc(const string &s) {
  int start = 0;
  for (;;) {
    int pos = s.find(":", start);
    string kvp;
    if (pos == string::npos) {
      kvp = s.substr(start);
      start = s.size();
    } else {
      kvp = s.substr(start, pos - start);
      start = pos + 1;
    }
    int q = kvp.find("=");
    if (q == string::npos) THROW("no '=' in Assoc");
    string key = kvp.substr(0, q);
    string value = kvp.substr(q + 1);
    (*this)[key] = value;
    if (start >= s.size()) break;
  }
}

void walk_params(Network net, ParamsFun f, const string &prefix) {
  for(auto it : net->parameters)
    f(prefix + "." + it.first, it.second);
  for(auto s : net->sub)
    walk_params(s, f, prefix + "." + s->kind);
}
void walk_states(Network net, StateFun f, const string &prefix) {
  for(auto it : net->states)
    f(prefix + "." + it.first, it.second);
  for(auto s : net->sub)
    walk_states(net, f, prefix + "." + s->kind);
}
void walk_networks(Network net, NetworkFun f, const string &prefix) {
  string nprefix = prefix + "." + net->kind;
  f(nprefix, net.get());
  for (int i = 0; i < net->sub.size(); i++) {
    walk_networks(net->sub[i], f, nprefix);
  }
}

void INetwork::gradientClipParameters(Float value) {
  for(auto it : parameters) {
    gradient_clip(*it.second, value);
  }
}
void INetwork::gradientClipStates(Float value) {
  for(auto it : states) {
    gradient_clip(*it.second, value);
  }
}
void INetwork::zeroGradsParameters() {
  for(auto it : parameters) {
    it.second->zeroGrad();
  }
}
void INetwork::zeroGradsStates() {
  for(auto it : states) {
    it.second->zeroGrad();
  }
}

map<string, ILayerFactory> layer_factories;

Network make_layer(const string &kind) {
  Network net;
  auto it = layer_factories.find(kind);
  if (it != layer_factories.end()) net.reset(it->second());
  return net;
}

Network layer(const string &kind, int ninput, int noutput, const Assoc &args,
              const Networks &subs) {
  Network net;
  auto it = layer_factories.find(kind);
  if (it != layer_factories.end()) {
    net.reset(it->second());
  } else {
    string accepted_layer_kinds = "";
    for (auto val : layer_factories) {
      accepted_layer_kinds += val.first;
      accepted_layer_kinds += ",";
    }
    THROW("unknown layer type:" + kind + ". Accepted layer kinds:" +
          accepted_layer_kinds);
  }

  for (auto it : args) {
    net->attr.set(it.first, it.second);
  }
  net->attr.set("ninput", ninput);
  net->attr.set("noutput", noutput);
  for (int i = 0; i < subs.size(); i++) {
    net->add(subs[i]);
    subs[i]->attr.super = &net->attr;
  }
  net->initialize();
  return net;
}

template <class T>
int register_layer(const char *name) {
  string s(name);
  layer_factories[s] = [s]() {
    T *result = new T();
    result->kind = s;
    return result;
  };
  return 0;
}
#define C(X, Y) X##Y
#define REGISTER(X) int C(status_, X) = register_layer<X>(#X);

Mat debugmat;

using namespace std;
using Eigen::Ref;

bool no_update = false;
bool verbose = false;

void set_inputs(INetwork *net, Sequence &inputs) {
  net->inputs.resize(inputs.size());
  for (int t = 0; t < net->inputs.size(); t++) {
    net->inputs[t] = inputs[t];
    net->inputs[t].zeroGrad();
  }
}
void set_targets(INetwork *net, Sequence &targets) {
  int N = net->outputs.size();
  assert(N == targets.size());
  assert(net->outputs.size() == N);
  for (int t = 0; t < N; t++) net->outputs[t].d = targets[t] - net->outputs[t];
}
void set_classes(INetwork *net, Classes &classes) {
  int N = net->outputs.size();
  assert(N == classes.size());
  assert(net->outputs.size() == N);
  for (int t = 0; t < N; t++) {
    net->outputs[t].d = -net->outputs[t];
    net->outputs[t].d(classes[t]) += 1;
  }
}

void INetwork::setLearningRate(Float lr, Float momentum) {
  attr.set("learning_rate", lr);
  attr.set("momentum", momentum);
}

Float INetwork::effective_lr() {
  Float lr = attr.get("learning_rate");
  string normalization = attr.get("normalization", "batch");
  if (normalization == "batch")
    lr /= fmax(1.0,nseq);
  else if (normalization == "len")
    lr /= fmax(1.0,nsteps);
  else if (normalization == "none") /* do nothing */
    ;
  else
    THROW("unknown normalization");
  return lr;
}

void INetwork::update() {
  Float lr = effective_lr();
  Float momentum = attr.get("momentum", 0.9);
  Float clip_at = attr.get("gradient_clip", 10.0);
  for(auto it : parameters)
    it.second->update(lr, momentum);
  for (int i = 0; i < sub.size(); i++)
    sub[i]->update();
  nseq = 0;
  nsteps = 0;
}

void Codec::set(const vector<int> &a) {
  codec = a;
  encoder.reset(new map<int, int>());
  for (int i = 0; i < codec.size(); i++) {
    encoder->insert(make_pair(codec[i], i));
  }
}

void Codec::encode(Classes &classes, const std::wstring &s) {
  classes.clear();
  for (int pos = 0; pos < s.size(); pos++) {
    unsigned c = s[pos];
    assert(encoder->count(c) > 0);
    c = (*encoder)[c];
    assert(c != 0);
    classes.push_back(c);
  }
}

wchar_t Codec::decode(int cls) { 
  return wchar_t(codec[cls]); 
}

std::wstring Codec::decode(Classes &classes) {
  std::wstring s;
  for (int i = 0; i < classes.size(); i++)
    s.push_back(wchar_t(codec[classes[i]]));
  return s;
}

void Codec::build(const vector<string> &fnames, const wstring &extra) {
  std::set<int> codes;
  codes.insert(0);
  for (auto c : extra) codes.insert(int(c));
  for (auto fname : fnames) {
    std::ifstream stream(fname);
    string line;
    wstring in, out;
    while (getline(stream, line)) {
      // skip blank lines and lines starting with a comment
      if (line.substr(0, 1) == "#") continue;
      if (line.size() == 0) continue;
      wstring s = utf8_to_utf32(line);
      for (auto c : s) codes.insert(int(c));
    }
  }
  vector<int> codec;
  for (auto c : codes) codec.push_back(c);
  for (int i = 1; i < codec.size(); i++) assert(codec[i] > codec[i - 1]);
  this->set(codec);
}

void network_info(Network net, string prefix) {
  string nprefix = prefix + "." + net->kind;
  Float learning_rate = net->attr.get("learning_rate");
  Float momentum = net->attr.get("momentum");
  cout << nprefix << ": " << learning_rate << " " << momentum << " ";
  cout << "in " << net->inputs.size() << " " << net->ninput() << " ";
  cout << "out " << net->outputs.size() << " " << net->noutput() << endl;
  for (auto s : net->sub) network_info(s, nprefix);
}

Sequence *get_state_by_name(Network net,string name) {
  Sequence *result = nullptr;
  walk_states(net, [&result, &name](const string &prefix, Sequence *s) {
    if (prefix == name) result = s;
  });
  return result;
}

template <class NONLIN>
struct Full : INetwork {
  Params W1;
  int nseq = 0;
  int nsteps = 0;
  Full() {
    ENROLL(W1);
  }
  void initialize() {
    int no = attr.get("noutput");
    int ni = attr.get("ninput");
    randinit(W1, no, ni + 1, 0.01);
    W1.zeroGrad();
  }
  int noutput() { return W1.rows(); }
  int ninput() { return W1.cols() - 1; }
  void forward() {
    outputs.resize(inputs.size(), W1.rows(), inputs.cols());
    for (int t = 0; t < inputs.size(); t++) {
      forward_full1<NONLIN>(outputs[t], W1, inputs[t]);
    }
  }
  void backward() {
    for (int t = outputs.size() - 1; t >= 0; t--) {
      backward_full1<NONLIN>(outputs[t], W1, inputs[t], 1000.0);
    }
    nsteps += outputs.size();
    nseq += 1;
  }
};

typedef Full<NoNonlin> LinearLayer;
REGISTER(LinearLayer);
typedef Full<SigmoidNonlin> SigmoidLayer;
REGISTER(SigmoidLayer);
typedef Full<TanhNonlin> TanhLayer;
REGISTER(TanhLayer);
typedef Full<ReluNonlin> ReluLayer;
REGISTER(ReluLayer);

struct SoftmaxLayer : INetwork {
  Params W1;
  int nsteps = 0;
  int nseq = 0;
  SoftmaxLayer() {
    ENROLL(W1);
  }
  void initialize() {
    int no = attr.get("noutput");
    int ni = attr.get("ninput");
    if (no < 2) THROW("Softmax requires no>=2");
    randinit(W1, no, ni + 1, 0.01);
    W1.zeroGrad();
  }
  int noutput() { return ROWS(W1); }
  int ninput() { return COLS(W1) - 1; }
  void postLoad() {
    W1.zeroGrad();
  }
  void forward() {
    outputs.resize(inputs.size(), W1.rows(), inputs.cols());
    for (int t = 0; t < inputs.size(); t++) {
      forward_softmax(outputs[t], W1, inputs[t]);
    }
  }
  void backward() {
    for (int t = outputs.size() - 1; t >= 0; t--) {
      backward_softmax(outputs[t], W1, inputs[t]);
    }
    nsteps += outputs.size();
    nseq += 1;
  }
};
REGISTER(SoftmaxLayer);

struct Stacked : INetwork {
  int noutput() { return sub[sub.size() - 1]->noutput(); }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(inputs.size() > 0);
    assert(inputs.rows() > 0);
    assert(inputs.cols() > 0);
    assert(sub.size() > 0);
    for (int n = 0; n < sub.size(); n++) {
      if (n == 0)
        sub[n]->inputs = inputs;
      else
        sub[n]->inputs = sub[n - 1]->outputs;
      sub[n]->forward();
    }
    outputs = sub[sub.size() - 1]->outputs;
    assert(outputs.size() == inputs.size());
    assert(outputs.cols() == inputs.cols());
  }
  void backward() {
    assert(outputs.size() > 0);
    assert(outputs.size() == inputs.size());
    for (int n = sub.size() - 1; n >= 0; n--) {
      if (n + 1 == sub.size())
        for (int t = 0; t < outputs.size(); t++)
          sub[n]->outputs[t].d = outputs[t].d;
      else
        for (int t = 0; t < sub[n + 1]->inputs.size(); t++)
          sub[n]->outputs[t].d = sub[n + 1]->inputs[t].d;
      sub[n]->backward();
    }
    for (int t = 0; t < sub[0]->inputs.size(); t++)
      inputs[t].d = sub[0]->inputs[t].d;
  }
};
REGISTER(Stacked);

struct Reversed : INetwork {
  int noutput() { return sub[0]->noutput(); }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(sub.size() == 1);
    Network net = sub[0];
    forward_reverse(net->inputs, inputs);
    net->forward();
    forward_reverse(outputs, net->outputs);
  }
  void backward() {
    Network net = sub[0];
    net->outputs.zeroGrad();
    backward_reverse(outputs, net->outputs);
    net->backward();
    outputs.zeroGrad();
    backward_reverse(net->inputs, inputs);
  }
};
REGISTER(Reversed);

struct Parallel : INetwork {
  int noutput() { 
    assert(sub[0]->noutput() > 0);
    assert(sub[1]->noutput() > 0);
    return sub[0]->noutput() + sub[1]->noutput(); 
  }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(sub.size() == 2);
    int N = inputs.size();
    sub[0]->inputs = inputs;
    sub[0]->forward();
    assert(sub[0]->outputs.size() == N);
    assert(sub[0]->outputs.cols() == inputs.cols());
    sub[1]->inputs = inputs;
    sub[1]->forward();
    assert(sub[1]->outputs.size() == N);
    assert(sub[1]->outputs.cols() == inputs.cols());
    outputs.resize(N, noutput(), inputs.cols());
    for (int t = 0; t < N; t++) {
      forward_stack(outputs[t], sub[0]->outputs[t], sub[1]->outputs[t]);
    }
  }
  void backward() {
    int N = outputs.size();
    sub[0]->outputs.zeroGrad();
    sub[1]->outputs.zeroGrad();
    for (int t = N-1; t >= 0; t--) {
      backward_stack(outputs[t], sub[0]->outputs[t], sub[1]->outputs[t]);
    }
    sub[0]->backward();
    sub[1]->backward();
    for (int t = 0; t < N; t++) {
      inputs[t].d = sub[0]->inputs[t].d;
      inputs[t].d += sub[1]->inputs[t].d;
    }
  }
};
REGISTER(Parallel);

template <class F = SigmoidNonlin, class G = TanhNonlin, class H = TanhNonlin>
struct GenericNPLSTM : INetwork {
#define WEIGHTS WGI, WGF, WGO, WCI
#define SEQUENCES gi, gf, go, ci, state
  Sequence source, SEQUENCES;
  Params WEIGHTS;
  Float gradient_clipping = 10.0;
  int ni, no, nf;
  int nsteps = 0;
  int nseq = 0;
  int noutput() { return no; }
  int ninput() { return ni; }
  GenericNPLSTM() {
    ENROLL(WGI, WGF, WGO, WCI);
    ENROLL(gi, gf, go, ci, state);
  }
  void initialize() {
    int ni = attr.get("ninput");
    int no = attr.get("noutput");
    int nf = 1 + ni + no;
    string mode = attr.get("weight_mode", "pos");
    float weight_dev = attr.get("weight_dev", 0.01);
    this->ni = ni;
    this->no = no;
    this->nf = nf;
    each([weight_dev, mode, no, nf](Params &w) {
      randinit(w, no, nf, weight_dev, mode);
      w.zeroGrad();
    }, WEIGHTS);
  }
  void postLoad() {
    no = ROWS(WGI);
    nf = COLS(WGI);
    assert(nf > no);
    ni = nf - no - 1;
  }
  void forward() {
    int N = inputs.size();
    int bs = inputs.cols();
    source.resize(N, nf, bs);
    state.resize(N, no, bs);
    gi.resize(N, no, bs);
    go.resize(N, no, bs);
    gf.resize(N, no, bs);
    ci.resize(N, no, bs);
    outputs.resize(N, no, bs);
    for (int t = 0; t < N; t++) {
      int bs = COLS(inputs[t]);
      forward_stack1(source[t], inputs[t], outputs, t - 1);
      forward_full<F>(gi[t], WGI, source[t]);
      forward_full<F>(gf[t], WGF, source[t]);
      forward_full<F>(go[t], WGO, source[t]);
      forward_full<G>(ci[t], WCI, source[t]);
      forward_statemem(state[t], ci[t], gi[t], state, t - 1, gf[t]);
      forward_nonlingate<H>(outputs[t], state[t], go[t]);
    }
  }
  void backward() {
    int N = inputs.size();
    int bs = outputs.cols();
    Sequence out;
    out.copy(outputs);
    for (int t = N - 1; t >= 0; t--) {
      backward_nonlingate<H>(out[t], state[t], go[t]);
      backward_statemem(state[t], ci[t], gi[t], state, t - 1, gf[t]);
      gradient_clip(state[t].d, gradient_clipping);
      backward_full<G>(ci[t], WCI, source[t], gradient_clipping);
      backward_full<F>(go[t], WGO, source[t], gradient_clipping);
      backward_full<F>(gf[t], WGF, source[t], gradient_clipping);
      backward_full<F>(gi[t], WGI, source[t], gradient_clipping);
      assert(gf[0].d.maxCoeff() == 0);
      backward_stack1(source[t], inputs[t], out, t - 1);
    }
    nsteps += N;
    nseq += 1;
  }
};
typedef GenericNPLSTM<> NPLSTM;
REGISTER(NPLSTM);

typedef GenericNPLSTM<SigmoidNonlin, TanhNonlin, NoNonlin> LINNPLSTM;
REGISTER(LINNPLSTM);

typedef GenericNPLSTM<SigmoidNonlin, ReluNonlin, TanhNonlin> RELUTANHNPLSTM;
REGISTER(RELUTANHNPLSTM);

typedef GenericNPLSTM<SigmoidNonlin, ReluNonlin, NoNonlin> RELUNPLSTM;
REGISTER(RELUNPLSTM);

typedef GenericNPLSTM<SigmoidNonlin, ReluNonlin, ReluNonlin> RELU2NPLSTM;
REGISTER(RELU2NPLSTM);

void save_net(const string &file, Network net) {
  save_as_proto(file, net.get());
}
Network load_net(const string &file) { return load_as_proto(file); }

}  // namespace ocropus
