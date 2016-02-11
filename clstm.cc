#include "clstm.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "clstm_compute.h"
#include "pstring.h"

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {

static vector<string> reported;

bool reported_params(const char *name) {
  string s(name);
  for (int i = 0; i < reported.size(); i++)
    if (reported[i] == s) return true;
  reported.push_back(s);
  return false;
}

void rinit(Params &m, int r, int c, Assoc &attr, string prefix = "") {
  m.resize(r, c);
  float s = attr.get(prefix + "init_scale", 0.01);
  string mode = attr.get(prefix + "init_mode", "negbiased");
  float offset = attr.get(prefix + "init_offset", 0.0);
  rinit(m, r, c, s, mode.c_str(), offset);
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
  for (auto &it : net->parameters) f(prefix + "." + it.first, it.second);
  for (auto &s : net->sub) walk_params(s, f, prefix + "." + s->kind);
}
void walk_states(Network net, StateFun f, const string &prefix, bool io) {
  if (io) {
    f(prefix + ".inputs", &net->inputs);
    f(prefix + ".outputs", &net->outputs);
  }
  for (auto &it : net->states) f(prefix + "." + it.first, it.second);
  for (auto &s : net->sub) walk_states(s, f, prefix + "." + s->kind, io);
}
void walk_networks(Network net, NetworkFun f, const string &prefix) {
  string nprefix = prefix + "." + net->kind;
  f(nprefix, net.get());
  for (int i = 0; i < net->sub.size(); i++) {
    walk_networks(net->sub[i], f, nprefix);
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

using namespace std;
using Eigen::Ref;

bool no_update = false;
bool verbose = false;

void set_inputs(Network net, Sequence &inputs) {
  // FIXME replace with operator in Sequence
  net->inputs.like(inputs);
  for (int t = 0; t < net->inputs.size(); t++) {
    net->inputs[t] = inputs[t];
    net->inputs[t].zeroGrad();
  }
}
void set_targets(Network net, Sequence &targets) {
  int N = net->outputs.size();
  assert(N == targets.size());
  assert(net->outputs.size() == N);
  // FIXME remove TensorMap operations
  for (int t = 0; t < N; t++)
    net->outputs[t].d() = targets[t].v() - net->outputs[t].v();
}
void set_classes(Network net, Classes &classes) {
  int N = net->outputs.size();
  assert(N == classes.size());
  assert(net->outputs.size() == N);
  // FIXME remove TensorMap operations
  for (int t = 0; t < N; t++) {
    net->outputs[t].d() = -net->outputs[t].v();
    net->outputs[t].d(classes[t], 0) += 1;
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
    lr /= fmax(1.0, nseq);
  else if (normalization == "len")
    lr /= fmax(1.0, nsteps);
  else if (normalization == "none") /* do nothing */
    ;
  else
    THROW("unknown normalization");
  return lr;
}

void INetwork::clearStates() {
  inputs.zero();
  outputs.zero();
  for (auto &it : states)
    it.second->zero();
}
void INetwork::clearStateDerivs() {
  // Clears the internal states (if any) and inputs.
  // Does not clear outputs (which is assumed to have
  // been set to the backwards gradient).
  inputs.zeroGrad();
  for (auto &it : states)
    it.second->zeroGrad();
}
void INetwork::clearWeightDerivs() {
  for (auto &it : parameters)
    it.second->zeroGrad();
}

void sgd_update(Network net) {
  Float lr = net->effective_lr();
  Float momentum = net->attr.get("momentum", 0.9);
  Float gc = net->attr.get("gradient_clip", 100.0);
  Float sgc = net->attr.get("state_gradient_clip", 100.0);
  for (auto it : net->parameters) {
    clip_gradient(*it.second, gc);
  }
  for (auto it : net->states) {
    Sequence &s = *it.second;
    for (int i = 0; i < s.size(); i++) clip_gradient(s[i], sgc);
  }
  for (auto it : net->parameters) sgd_update(*it.second, lr, momentum);
  for (int i = 0; i < net->sub.size(); i++) sgd_update(net->sub[i]);
  net->nseq = 0;
  net->nsteps = 0;
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

wchar_t Codec::decode(int cls) { return wchar_t(codec[cls]); }

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

struct Info {
  double min = INFINITY;
  double max = -INFINITY;
  double sum = 0.0;
  double count = 0.0;
  void operator+=(double x) {
    if (x < min) min = x;
    if (x > max) max = x;
    sum += x;
    count += 1;
  }
  string info() {
    return "[" + to_string(min) + "|" + to_string(sum / count) + "|" +
           to_string(max) + ":" + to_string(int(count)) + "]";
  }
};

string info(Sequence &s) {
  Info info, dinfo;
  for (int t = 0; t < s.size(); t++) {
    for (int i = 0; i < s.rows(); i++) {
      for (int b = 0; b < s.cols(); b++) {
        info += s[t].v(i, b);
        dinfo += s[t].d(i, b);
      }
    }
  }
  return "Seq:" + info.info() + dinfo.info();
  ;
}

string info(Batch &s) {
  Info info, dinfo;
  for (int i = 0; i < s.rows(); i++) {
    for (int b = 0; b < s.cols(); b++) {
      info += s.v(i, b);
      dinfo += s.d(i, b);
    }
  }
  return "Bat:" + info.info() + dinfo.info();
  ;
}

void network_detail(Network net, string prefix) {
  string nprefix = prefix + "." + net->kind;
  Float learning_rate = net->attr.get("learning_rate");
  Float momentum = net->attr.get("momentum");
  cout << nprefix << " <<<" << learning_rate << " " << momentum << " ";
  cout << "in " << net->inputs.size() << " " << net->ninput() << " ";
  cout << "out " << net->outputs.size() << " " << net->noutput() << ">>>"
       << endl;
  for (auto p : net->parameters) {
    cout << nprefix << "    " << p.first << " " << p.second->rows() << " "
         << p.second->cols() << " " << info(*p.second) << endl;
  }
  auto show = [&](const string &s, Sequence *p) {
    cout << nprefix << "    " << s << " " << p->size() << " " << p->rows()
         << " " << p->cols() << " " << info(*p) << endl;
  };
  show("inputs", &net->inputs);
  show("outputs", &net->outputs);
  for (auto p : net->states) {
    show(p.first, p.second);
  }
  for (auto s : net->sub) network_detail(s, nprefix);
}

Sequence *get_state_by_name(Network net, string name) {
  Sequence *result = nullptr;
  walk_states(net, [&result, &name](const string &prefix, Sequence *s) {
    if (prefix == name) result = s;
  });
  return result;
}

template <int NONLIN>
struct Full : INetwork {
  Params W1;
  int nseq = 0;
  int nsteps = 0;
  Full() { ENROLL(W1); }
  void initialize() {
    int no = attr.get("noutput");
    int ni = attr.get("ninput");
    rinit(W1, no, ni + 1, attr);
  }
  int noutput() { return W1.rows(); }
  int ninput() { return W1.cols() - 1; }
  void forward() {
    outputs.resize(inputs.size(), W1.rows(), inputs.cols());
    for (int t = 0; t < inputs.size(); t++) {
      forward_full1(outputs[t], W1, inputs[t], NONLIN);
    }
  }
  void backward() {
    for (int t = outputs.size() - 1; t >= 0; t--) {
      backward_full1(outputs[t], W1, inputs[t], NONLIN);
    }
    nsteps += outputs.size();
    nseq += 1;
  }
};

typedef Full<LIN> LinearLayer;
REGISTER(LinearLayer);
typedef Full<SIG> SigmoidLayer;
REGISTER(SigmoidLayer);
typedef Full<TANH> TanhLayer;
REGISTER(TanhLayer);
typedef Full<RELU> ReluLayer;
REGISTER(ReluLayer);

struct SoftmaxLayer : INetwork {
  Params W1;
  int nsteps = 0;
  int nseq = 0;
  SoftmaxLayer() { ENROLL(W1); }
  void initialize() {
    int no = attr.get("noutput");
    int ni = attr.get("ninput");
    if (no < 2) THROW("Softmax requires no>=2");
    rinit(W1, no, ni + 1, attr);
  }
  int noutput() { return ROWS(W1); }
  int ninput() { return COLS(W1) - 1; }
  void postLoad() { W1.zeroGrad(); }
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
    net->inputs.like(inputs);
    forward_reverse(net->inputs, inputs);
    net->forward();
    outputs.like(net->outputs);
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

struct Btswitch : INetwork {
  void forward() {
    outputs.resize(inputs.cols(), inputs.rows(), inputs.size());
    forward_btswitch(outputs, inputs);
  }
  void backward() { backward_btswitch(outputs, inputs); }
};
REGISTER(Btswitch);

struct Batchstack : INetwork {
  int pre = 1;
  int post = 1;
  void initialize() {
    pre = attr.get("pre", 1);
    post = attr.get("post", pre);
  }
  void forward() {
    outputs.resize(inputs.size(), inputs.rows() * (pre + post + 1),
                   inputs.cols());
    forward_batchstack(outputs, inputs);
  }
  void backward() { backward_batchstack(outputs, inputs); }
};
REGISTER(Batchstack);

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
    for (int t = N - 1; t >= 0; t--) {
      backward_stack(outputs[t], sub[0]->outputs[t], sub[1]->outputs[t]);
    }
    sub[0]->backward();
    sub[1]->backward();
    for (int t = 0; t < N; t++) {
      inputs[t].d() = sub[0]->inputs[t].d();
      inputs[t].d() += sub[1]->inputs[t].d();
    }
  }
};
REGISTER(Parallel);

template <int F = SIG, int G = TANH, int H = TANH>
struct GenericNPLSTM : INetwork {
#define WEIGHTS WGI, WGF, WGO, WCI
#define SEQUENCES gi, gf, go, ci, state
  Sequence source, SEQUENCES;
  Params WEIGHTS;
  int ni, no, nf;
  int nsteps = 0;
  int nseq = 0;
  int noutput() { return no; }
  int ninput() { return ni; }
  int gpu = -1;
  GenericNPLSTM() {
    ENROLL(WGI, WGF, WGO, WCI);
    ENROLL(gi, gf, go, ci, state, source);
  }
  void initialize() {
    int ni = attr.get("ninput");
    int no = attr.get("noutput");
    int nf = ni + no;
    string mode = attr.get("weight_mode", "pos");
    // float weight_dev = attr.get("weight_dev", 0.01);
    this->ni = ni;
    this->no = no;
    this->nf = nf;
    gpu = attr.get("gpu", -1);
    if (gpu >= 0) {
      cerr << "LSTM gpu = " << gpu << "\n";
      WGI.setGpu(gpu);
      WGF.setGpu(gpu);
      WGO.setGpu(gpu);
      WCI.setGpu(gpu);
      gi.setGpu(gpu);
      gf.setGpu(gpu);
      go.setGpu(gpu);
      ci.setGpu(gpu);
      state.setGpu(gpu);
      source.setGpu(gpu);
      inputs.setGpu(gpu);
      outputs.setGpu(gpu);
    }
    assert(WGI.v.getGpu() == gpu);
    rinit(WGI, no, nf + 1, attr);
    rinit(WGF, no, nf + 1, attr);
    rinit(WGO, no, nf + 1, attr);
    rinit(WCI, no, nf + 1, attr);
    assert(WGI.v.getGpu() == gpu);
  }
  void postLoad() {
    no = WGI.rows();
    nf = WGI.cols() - 1;
    ni = nf - no;
    assert(nf > no);
  }
  void forward() {
    assert(inputs.getGpu() == gpu);
    int N = inputs.size();
    int bs = inputs.cols();
    source.resize(N, nf, bs);
    state.resize(N, no, bs);
    gi.resize(N, no, bs);
    go.resize(N, no, bs);
    gf.resize(N, no, bs);
    ci.resize(N, no, bs);
    outputs.resize(N, no, bs);
    assert(inputs.getGpu() == gpu);
    for (int t = 0; t < N; t++) {
      forward_stack_delay(source[t], inputs[t], outputs, t - 1);
      forward_full1(gi[t], WGI, source[t], F);
      forward_full1(gf[t], WGF, source[t], F);
      forward_full1(go[t], WGO, source[t], F);
      forward_full1(ci[t], WCI, source[t], G);
      forward_statemem(state[t], ci[t], gi[t], state, t - 1, gf[t]);
      forward_nonlingate(outputs[t], state[t], go[t], H);
    }
  }
  void backward() {
    clearStateDerivs();
    int N = inputs.size();
    // int bs = outputs.cols();
    Sequence out;
    out.setGpu(gpu);
    out = outputs;
    for (int t = N - 1; t >= 0; t--) {
      assert(!anynan(source[t]));
      backward_nonlingate(out[t], state[t], go[t], H);
      assert(!anynan(source[t]));
      backward_statemem(state[t], ci[t], gi[t], state, t - 1, gf[t]);
      assert(!anynan(source[t]));
      backward_full1(ci[t], WCI, source[t], G);
      assert(!anynan(source[t]));
      backward_full1(go[t], WGO, source[t], F);
      assert(!anynan(source[t]));
      backward_full1(gf[t], WGF, source[t], F);
      assert(!anynan(source[t]));
      backward_full1(gi[t], WGI, source[t], F);
      assert(!anynan(out));
      assert(!anynan(source[t]));
      backward_stack_delay(source[t], inputs[t], out, t - 1);
      assert(!anynan(source[t]));
      if(anynan(inputs[t])) {
        print("oops", t);
        assert(!anynan(inputs[t]));
      }
    }
    nsteps += N;
    nseq += 1;
  }
};
typedef GenericNPLSTM<> NPLSTM;
REGISTER(NPLSTM);

typedef GenericNPLSTM<SIG, TANH, LIN> LINNPLSTM;
REGISTER(LINNPLSTM);

typedef GenericNPLSTM<SIG, RELU, TANH> RELUTANHNPLSTM;
REGISTER(RELUTANHNPLSTM);

typedef GenericNPLSTM<SIG, RELU, LIN> RELUNPLSTM;
REGISTER(RELUNPLSTM);

typedef GenericNPLSTM<SIG, RELU, RELU> RELU2NPLSTM;
REGISTER(RELU2NPLSTM);

bool maybe_save_net(const string &file, Network net) {
  return save_as_proto(file, net.get());
}
Network maybe_load_net(const string &file) { return load_as_proto(file); }
void save_net(const string &file, Network net) {
  if (save_as_proto(file, net.get())) return;
  THROW("error saving network");
}
Network load_net(const string &file) {
  Network result = load_as_proto(file);
  if (!result) THROW("error loading network");
  return result;
}

void set_inputs(Network net, TensorMap2 inputs) {
  int N = inputs.dimension(0);
  int d = inputs.dimension(1);
  net->inputs.resize(N, d, 1);
  for (int t = 0; t < N; t++)
    for (int i = 0; i < d; i++) net->inputs[t].v(i, 0) = inputs(t, i);
}
void set_targets(Network net, TensorMap2 targets) {
  int N = targets.dimension(0);
  int d = targets.dimension(1);
  for (int t = 0; t < N; t++)
    for (int i = 0; i < d; i++) net->outputs[t].d(i, 0) = targets(t, i);
  for (int t = 0; t < net->outputs.size(); t++)
    net->outputs[t].d() -= net->outputs[t].v();
}
void set_targets_accelerated(Network net, Tensor<float, 2> &targets) {
  THROW("unimplemented");
}
void set_classes(Network net, Tensor<int, 1> &targets) {
  THROW("unimplemented");
}

static void get_allparams(vector<vector<Params *>> &allparams,
                          vector<Network> &networks) {
  allparams.resize(networks.size());
  for (int i = 0; i < allparams.size(); i++) {
    Network net = networks[i];
    walk_params(net, [i, &allparams](const string &s, Params *p) {
      allparams[i].push_back(p);
    });
    assert(allparams[i].size() == allparams[0].size());
  }
}

void distribute_weights(vector<Network> &networks, int from) {
  vector<vector<Params *>> allparams;
  get_allparams(allparams, networks);
  int n = allparams.size();
  int m = allparams[0].size();
  for (int i = 0; i < n; i++) {
    if (i == from) continue;
    for (int j = 0; j < m; j++) {
      allparams[i][j]->v() = allparams[from][j]->v();
    }
  }
}

void share_deltas(vector<Network> &networks) {
  vector<vector<Params *>> allparams;
  get_allparams(allparams, networks);
  int n = allparams.size();
  int m = allparams[0].size();
  for (int i = 1; i < n; i++) {
    for (int j = 0; j < m; j++) {
      allparams[0][j]->d() += allparams[i][j]->d();
    }
    for (int j = 0; j < m; j++) {
      allparams[i][j]->d() = allparams[0][j]->d();
    }
  }
}

void average_weights(vector<Network> &networks) {
  vector<vector<Params *>> allparams;
  get_allparams(allparams, networks);
  int n = allparams.size();
  int m = allparams[0].size();
  for (int i = 1; i < n; i++) {
    for (int j = 0; j < m; j++) {
      allparams[0][j]->v() += allparams[i][j]->v();
    }
  }
  for (int j = 0; j < m; j++) {
    allparams[0][j]->v() = allparams[0][j]->v() * Float(1.0 / n);
  }
  distribute_weights(networks);
}

int n_states(Network net) {
  int total = 0;
  walk_states(  //
      net, [&](const string &,
               Sequence *p) { total += p->size() * p->rows() * p->cols() + 4; },
      "", true);
  return total;
}

void get_states(Network net, Float *data, int total, int gpu) {
  int index = 0;
  walk_states(  //
      net,
      [&](const string &, Sequence *p) {
        data[index++] = 999999;
        data[index++] = p->size();
        data[index++] = p->rows();
        data[index++] = p->cols();
        if (index + p->size() * p->rows() * p->cols() > total)
          THROW("size mismatch in get_states");
        for (int t = 0; t < p->size(); t++)
          for (int i = 0; i < p->rows(); i++)
            for (int b = 0; b < p->cols(); b++) data[index++] = (*p)[t].v(i, b);
      },
      "", true);
  if (index != total) THROW("size mismatch in get_states");
}

void set_states(Network net, const Float *data, int total, int gpu) {
  int index = 0;
  walk_states(  //
      net,
      [&](const string &name, Sequence *p) {
        int magic = int(data[index++]);
        assert(magic == 999999);
        if (magic != 999999) return;
        int size = int(data[index++]);
        int rows = int(data[index++]);
        int cols = int(data[index++]);
        if (index + size * rows * cols > total)
          THROW("size mismatch in set_states");
        p->resize(size, rows, cols);
        for (int t = 0; t < size; t++)
          for (int i = 0; i < rows; i++)
            for (int b = 0; b < cols; b++) (*p)[t].v(i, b) = data[index++];
      },
      "", true);
  if (total != index) THROW("size mismatch in set_states");
}

void clear_states(Network net) {
  int index = 0;
  walk_states(  //
      net, [&](const string &, Sequence *p) { p->clear(); }, "", true);
}

void invalidate_state_derivs(Network net) {
  walk_states(  //
      net,
      [&](const string &, Sequence *p) {
        for (int t = 0; t < p->size(); t++)
          for (int i = 0; i < p->rows(); i++)
            for (int b = 0; b < p->cols(); b++) (*p)[t].d(i, b) = NAN;
      },
      "", true);
}

void clear_state_derivs(Network net) {
  walk_states(  //
      net,
      [&](const string &, Sequence *p) {
        for (int t = 0; t < p->size(); t++)
          for (int i = 0; i < p->rows(); i++)
            for (int b = 0; b < p->cols(); b++) (*p)[t].d(i, b) = 0;
      },
      "", true);
}

int n_params(Network net) {
  int total = 0;
  walk_params(net,
              [&](const string &, Params *p) { total += p->v.total_size(); });
  return total;
}

void share_params(Network net, Float *params, int total, int gpu) {
  int index = 0;
  walk_params(net, [&](const string &, Params *p) {
    int n = p->v.rows();
    int m = p->v.cols();
    if (index + p->v.total_size() > total) THROW("share_params size mismatch");
    p->v.displaceTo(params + index, n, m, gpu);
    index += p->v.total_size();
  });
  if (index != total) THROW("share_params size mismatch");
}

void set_params(Network net, const Float *params, int total, int gpu) {
  assert(gpu < 0);
  int index = 0;
  walk_params(net, [&](const string &, Params *p) {
    int n = p->v.rows();
    int m = p->v.cols();
    int nbytes = p->v.total_size() * sizeof(Float);
    if (index + p->v.total_size() > total) THROW("get_params size mismatch");
    memcpy(p->v.ptr, params + index, nbytes);
    index += p->v.total_size();
  });
  if (index != total) THROW("get_params size mismatch");
}

void get_params(Network net, Float *params, int total, int gpu) {
  assert(gpu < 0);
  int index = 0;
  walk_params(net, [&](const string &, Params *p) {
    int n = p->v.rows();
    int m = p->v.cols();
    int nbytes = p->v.total_size() * sizeof(Float);
    if (index + p->v.total_size() > total) THROW("get_params size mismatch");
    memcpy(params + index, p->v.ptr, nbytes);
    index += p->v.total_size();
  });
  if (index != total) THROW("get_params size mismatch");
}

void clear_derivs(Network net) {
  walk_params(net, [&](const string &, Params *p) { p->d.setZero(); });
  walk_states(net, [&](const string &, Sequence *p) {
    for (int t = 0; t < p->size(); t++) (*p)[t].d.setZero();
  });
}

void get_derivs(Network net, Float *params, int total, int gpu) {
  assert(gpu < 0);
  int index = 0;
  walk_params(net, [&](const string &, Params *p) {
    int n = p->v.rows();
    int m = p->v.cols();
    int nbytes = p->v.total_size() * sizeof(Float);
    memcpy(params + index, p->d.ptr, nbytes);
    index += p->v.total_size();
  });
  if (index != total) THROW("get_derivs size mismatch");
}

void set_derivs(Network net, Float *params, int total, int gpu) {
  assert(gpu < 0);
  int index = 0;
  walk_params(net, [&](const string &, Params *p) {
    int n = p->v.rows();
    int m = p->v.cols();
    int nbytes = p->v.total_size() * sizeof(Float);
    memcpy(p->d.ptr, params + index, nbytes);
    index += p->v.total_size();
  });
  if (index != total) THROW("set_derivs size mismatch");
}
}  // namespace ocropus
