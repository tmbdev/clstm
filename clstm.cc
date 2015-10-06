#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>

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
    net->attributes[it.first] = it.second;
  }
  net->attributes["ninput"] = std::to_string(ninput);
  net->attributes["noutput"] = std::to_string(noutput);
  for (int i = 0; i < subs.size(); i++) net->sub.push_back(subs[i]);
  net->initialize();
  return net;
}

template <class T>
int register_layer(const char *name) {
  T *net = new T();
  string kind = net->kind();
  delete net;
  string s(name);
  layer_factories[s] = []() { return new T(); };
  layer_factories[kind] = []() { return new T(); };
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
void set_targets_accelerated(INetwork *net, Sequence &targets) {
  Float lo = 1e-5;
  assert(net->outputs.size() == targets.size());
  int N = net->outputs.size();
  assert(net->outputs.size() == N);
  for (int t = 0; t < N; t++) {
    net->outputs[t].d = -net->outputs[t];
    for (int i = 0; i < ROWS(targets[t]); i++) {
      for (int b = 0; b < COLS(targets[t]); b++) {
        // only allow binary classification
        assert(fabs(targets[t](i, b) - 0) < 1e-5 ||
               fabs(targets[t](i, b) - 1) < 1e-5);
        if (targets[t](i, b) > 0.5) {
          net->outputs[t].d(i, b) = 1.0 / fmax(lo, net->outputs[t](i, b));
        }
      }
    }
  }
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
void train(INetwork *net, Sequence &xs, Sequence &targets) {
  assert(xs.size() > 0);
  assert(xs.size() == targets.size());
  net->inputs = xs;
  net->forward();
  set_targets(net, targets);
  net->backward();
  net->update();
}
void ctrain(INetwork *net, Sequence &xs, Classes &cs) {
  net->inputs = xs;
  net->forward();
  int len = net->outputs.size();
  assert(len > 0);
  int dim = net->outputs[0].size();
  assert(dim > 0);
  assert(net->outputs.size() == len);
  if (dim == 1) {
    for (int t = 0; t < len; t++)
      net->outputs[t].d(0) =
          cs[t] ? 1.0 - net->outputs[t](0) : -net->outputs[t](0);
  } else {
    for (int t = 0; t < len; t++) {
      net->outputs[t].d = -net->outputs[t];
      int c = cs[t];
      net->outputs[t].d(c) = 1 - net->outputs[t](c);
    }
  }
  net->backward();
  net->update();
}

void ctrain_accelerated(INetwork *net, Sequence &xs, Classes &cs, Float lo) {
  net->inputs = xs;
  net->forward();
  int len = net->outputs.size();
  assert(len > 0);
  int dim = net->outputs[0].size();
  assert(dim > 0);
  if (dim == 1) {
    for (int t = 0; t < len; t++) {
      if (cs[t] == 0)
        net->outputs[t].d(0) = -1.0 / fmax(lo, 1.0 - net->outputs[t](0));
      else
        net->outputs[t].d(0) = 1.0 / fmax(lo, net->outputs[t](0));
    }
  } else {
    for (int t = 0; t < len; t++) {
      net->outputs[t].d = -net->outputs[t];
      int c = cs[t];
      net->outputs[t].d(c) = 1.0 / fmax(lo, net->outputs[t](c));
    }
  }
  net->backward();
  net->update();
}

void cpred(INetwork *net, Classes &preds, Sequence &xs) {
  int N = xs.size();
  assert(COLS(xs[0]) == 0);
  net->inputs = xs;
  preds.resize(N);
  net->forward();
  assert(net->outputs.size() == N);
  for (int t = 0; t < N; t++) {
    int index = -1;
    net->outputs[t].col(0).maxCoeff(&index);
    preds[t] = index;
  }
}

void INetwork::makeEncoders() {
  encoder.reset(new map<int, int>());
  for (int i = 0; i < codec.size(); i++) {
    encoder->insert(make_pair(codec[i], i));
  }
  iencoder.reset(new map<int, int>());
  for (int i = 0; i < icodec.size(); i++) {
    iencoder->insert(make_pair(icodec[i], i));
  }
}

void INetwork::encode(Classes &classes, const std::wstring &s) {
  if (!encoder) makeEncoders();
  classes.clear();
  for (int pos = 0; pos < s.size(); pos++) {
    unsigned c = s[pos];
    assert(encoder->count(c) > 0);
    c = (*encoder)[c];
    assert(c != 0);
    classes.push_back(c);
  }
}
void INetwork::iencode(Classes &classes, const std::wstring &s) {
  if (!iencoder) makeEncoders();
  classes.clear();
  for (int pos = 0; pos < s.size(); pos++) {
    int c = (*iencoder)[int(s[pos])];
    classes.push_back(c);
  }
}
wchar_t INetwork::decode(int cls) { return wchar_t(codec[cls]); }
wchar_t INetwork::idecode(int cls) { return wchar_t(icodec[cls]); }
std::wstring INetwork::decode(Classes &classes) {
  std::wstring s;
  for (int i = 0; i < classes.size(); i++)
    s.push_back(wchar_t(codec[classes[i]]));
  return s;
}
std::wstring INetwork::idecode(Classes &classes) {
  std::wstring s;
  for (int i = 0; i < classes.size(); i++)
    s.push_back(wchar_t(icodec[classes[i]]));
  return s;
}

void INetwork::info(string prefix) {
  string nprefix = prefix + "." + name;
  cout << nprefix << ": " << learning_rate << " " << momentum << " ";
  cout << "in " << inputs.size() << " " << ninput() << " ";
  cout << "out " << outputs.size() << " " << noutput() << endl;
  for (auto s : sub) s->info(nprefix);
}

void INetwork::weights(const string &prefix, WeightFun f) {
  string nprefix = prefix + "." + name;
  myweights(nprefix, f);
  for (int i = 0; i < sub.size(); i++) {
    sub[i]->weights(nprefix + "." + to_string(i), f);
  }
}

void INetwork::params(const string &prefix, ParamsFun f) {
  string nprefix = prefix + "." + name;
  myparams(nprefix, f);
  for (int i = 0; i < sub.size(); i++) {
    sub[i]->params(nprefix + "." + to_string(i), f);
  }
}

void INetwork::states(const string &prefix, StateFun f) {
  string nprefix = prefix + "." + name;
  f(nprefix + ".inputs", &inputs);
  f(nprefix + ".outputs", &outputs);
  mystates(nprefix, f);
  for (int i = 0; i < sub.size(); i++) {
    sub[i]->states(nprefix + "." + to_string(i), f);
  }
}

void INetwork::networks(const string &prefix,
                        function<void(string, INetwork *)> f) {
  string nprefix = prefix + "." + kind();
  f(nprefix, this);
  for (int i = 0; i < sub.size(); i++) {
    sub[i]->networks(nprefix, f);
  }
}

Sequence *INetwork::getState(string name) {
  Sequence *result = nullptr;
  states("", [&result, &name](const string &prefix, Sequence *s) {
    if (prefix == name) result = s;
  });
  return result;
}

struct NetworkBase : INetwork {
  Float error2(Sequence &xs, Sequence &targets) {
    inputs = xs;
    forward();
    Float total = 0.0;
    for (int t = 0; t < outputs.size(); t++) {
      Vec delta = targets[t] - outputs[t];
      total += delta.array().square().sum();
      outputs[t].d = delta;
    }
    backward();
    update();
    return total;
  }
};

template <class NONLIN>
struct Full : NetworkBase {
  Params W1;
  int nseq = 0;
  int nsteps = 0;
  string mykind = string("Full_") + NONLIN::kind;
  Full() { name = string("full_") + NONLIN::name; }
  const char *kind() { return mykind.c_str(); }
  int noutput() { return ROWS(W1); }
  int ninput() { return COLS(W1) - 1; }
  void initialize() {
    int no = irequire("noutput");
    int ni = irequire("ninput");
    randinit(W1, no, ni + 1, 0.01);
    W1.zeroGrad();
  }
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
  void update() {
    float lr = learning_rate;
    if (normalization == NORM_BATCH)
      lr /= nseq;
    else if (normalization == NORM_LEN)
      lr /= nsteps;
    else if (normalization == NORM_NONE) /* do nothing */
      ;
    else
      THROW("unknown normalization");
    W1 += lr * W1.d;
    W1.d *= momentum;
    nsteps = 0;
    nseq = 0;
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".W1", &W1, (Mat *)0);
  }
  void myparams(const string &prefix, ParamsFun f) { f(prefix + ".W1", &W1); }
};

typedef Full<NoNonlin> LinearLayer;
REGISTER(LinearLayer);
typedef Full<SigmoidNonlin> SigmoidLayer;
REGISTER(SigmoidLayer);
typedef Full<TanhNonlin> TanhLayer;
REGISTER(TanhLayer);
typedef Full<ReluNonlin> ReluLayer;
REGISTER(ReluLayer);

struct SoftmaxLayer : NetworkBase {
  Params W1;
  int nsteps = 0;
  int nseq = 0;
  SoftmaxLayer() { name = "softmax"; }
  const char *kind() { return "SoftmaxLayer"; }
  int noutput() { return ROWS(W1); }
  int ninput() { return COLS(W1) - 1; }
  void initialize() {
    int no = irequire("noutput");
    int ni = irequire("ninput");
    if (no < 2) THROW("Softmax requires no>=2");
    randinit(W1, no, ni + 1, 0.01);
    clearUpdates();
  }
  void clearUpdates() { W1.zeroGrad(); }
  void postLoad() {
    W1.zeroGrad();
    makeEncoders();
  }
  void forward() {
    int nsteps = inputs.size();
    int no = ROWS(W1), bs = COLS(inputs[0]);
    outputs.resize(nsteps, no, bs);
    for (int t = 0; t < inputs.size(); t++) {
      outputs[t] = MAPFUN(HOMDOT(W1, inputs[t]), limexp);
      for (int b = 0; b < COLS(outputs[t]); b++) {
        Float total = fmax(SUMREDUCE(COL(outputs[t], b)), 1e-9);
        COL(outputs[t], b) /= total;
      }
    }
  }
  void backward() {
    for (int t = outputs.size() - 1; t >= 0; t--) {
      inputs[t].d = MATMUL_TR(CBUTFIRST(W1), outputs[t].d);
    }
    int bs = COLS(inputs[0]);
    for (int t = 0; t < outputs.size(); t++) {
      auto d_W = CBUTFIRST(W1.d);
      d_W += MATMUL_RT(outputs[t].d, inputs[t]);
      auto d_w = CFIRST(W1.d);
      for (int b = 0; b < bs; b++) d_w += COL(outputs[t].d, b);
    }
    nsteps += outputs.size();
    nseq += 1;
  }
  void update() {
    float lr = learning_rate;
    if (normalization == NORM_BATCH)
      lr /= nseq;
    else if (normalization == NORM_LEN)
      lr /= nsteps;
    else if (normalization == NORM_NONE) /* do nothing */
      ;
    else
      THROW("unknown normalization");
    W1 += lr * W1.d;
    W1.d *= momentum;
    nsteps = 0;
    nseq = 0;
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".W1", &W1, &W1.d);
  }
  void myparams(const string &prefix, ParamsFun f) { f(prefix + ".W1", &W1); }
};
REGISTER(SoftmaxLayer);

struct Stacked : NetworkBase {
  Stacked() { name = "stacked"; }
  const char *kind() { return "Stacked"; }
  int noutput() { return sub[sub.size() - 1]->noutput(); }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(inputs.size() > 0);
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
  void update() {
    for (int i = 0; i < sub.size(); i++) sub[i]->update();
  }
};
REGISTER(Stacked);

struct Reversed : NetworkBase {
  Reversed() { name = "reversed"; }
  const char *kind() { return "Reversed"; }
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
  void update() { sub[0]->update(); }
};
REGISTER(Reversed);

struct Parallel : NetworkBase {
  Parallel() { name = "parallel"; }
  const char *kind() { return "Parallel"; }
  int noutput() { return sub[0]->noutput() + sub[1]->noutput(); }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(sub.size() == 2);
    int N = inputs.size();
    outputs.resize(N, noutput(), inputs.cols());
    sub[0]->inputs = inputs;
    sub[1]->inputs = inputs;
    sub[0]->forward();
    sub[1]->forward();
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
  void update() {
    for (int i = 0; i < sub.size(); i++) sub[i]->update();
  }
};
REGISTER(Parallel);

template <class F = SigmoidNonlin, class G = TanhNonlin, class H = TanhNonlin>
struct GenericNPLSTM : NetworkBase {
#define WEIGHTS WGI, WGF, WGO, WCI
#define SEQUENCES gi, gf, go, ci, state
  Sequence source, SEQUENCES;
  Params WEIGHTS;
  Float gradient_clipping = 10.0;
  int ni, no, nf;
  int nsteps = 0;
  int nseq = 0;
  string mykind = string("NPLSTM_") + F::kind + G::kind + H::kind;
  GenericNPLSTM() { name = "lstm"; }
  const char *kind() { return mykind.c_str(); }
  int noutput() { return no; }
  int ninput() { return ni; }
  void postLoad() {
    no = ROWS(WGI);
    nf = COLS(WGI);
    assert(nf > no);
    ni = nf - no - 1;
    clearUpdates();
  }
  void initialize() {
    int ni = irequire("ninput");
    int no = irequire("noutput");
    int nf = 1 + ni + no;
    string mode = attr("weight_mode", "pos");
    float weight_dev = dattr("weight_dev", 0.01);
    this->ni = ni;
    this->no = no;
    this->nf = nf;
    each([weight_dev, mode, no, nf](Mat &w) {
      randinit(w, no, nf, weight_dev, mode);
    }, WEIGHTS);
#if 0
        float gf_mean = dattr("gf_mean", 0.0);
        float gf_dev = dattr("gf_dev", 0.01);
        Vec offset;
        randinit(offset, no, gf_dev, mode);
        offset.array() += gf_mean;
        COL(WGF, 0) = offset;
#endif
    clearUpdates();
  }
  void clearUpdates() {
    each([this](Params &w) { w.d = Mat::Zero(no, nf); }, WEIGHTS);
  }
  void resize(int N) {
    each([N](Sequence &s) {
      s.resize(N);
      for (int t = 0; t < N; t++) s[t].setConstant(NAN);
      for (int t = 0; t < N; t++) s[t].d.setConstant(NAN);
    }, source, outputs, SEQUENCES);
    assert(source.size() == N);
    assert(gi.size() == N);
    assert(go.size() == N);
  }
  void forward() {
    int N = inputs.size();
    int bs = inputs.cols();
    resize(N);
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
    each([](Sequence &s) { s.zeroGrad(); }, source, inputs, state, gi, go, gf,
         ci);

    for (int t = N - 1; t >= 0; t--) {
      backward_nonlingate<H>(out[t], state[t], go[t]);
      backward_statemem(state[t], ci[t], gi[t], state, t - 1, gf[t]);
      gradient_clip(state[t].d, gradient_clipping);
      backward_full<F>(gi[t], WGI, source[t], gradient_clipping);
      assert(gf[0].d.maxCoeff() == 0);
      backward_full<F>(gf[t], WGF, source[t], gradient_clipping);
      backward_full<F>(go[t], WGO, source[t], gradient_clipping);
      backward_full<G>(ci[t], WCI, source[t], gradient_clipping);
      backward_stack1(source[t], inputs[t], out, t - 1);
    }
    nsteps += N;
    nseq += 1;
  }
  void update() {
    float lr = learning_rate;
    if (normalization == NORM_BATCH)
      lr /= nseq;
    else if (normalization == NORM_LEN)
      lr /= nsteps;
    else if (normalization == NORM_NONE) /* do nothing */
      ;
    else
      THROW("unknown normalization");
    each([this, lr](Params &W) {
      W += lr * W.d;
      W.d *= momentum;
    }, WGI, WGF, WGO, WCI);
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".WGI", &WGI, &WGI.d);
    f(prefix + ".WGF", &WGF, &WGF.d);
    f(prefix + ".WGO", &WGO, &WGO.d);
    f(prefix + ".WCI", &WCI, &WCI.d);
  }
  void myparams(const string &prefix, ParamsFun f) {
    f(prefix + ".WGI", &WGI);
    f(prefix + ".WGF", &WGF);
    f(prefix + ".WGO", &WGO);
    f(prefix + ".WCI", &WCI);
  }
  virtual void mystates(const string &prefix, StateFun f) {
    f(prefix + ".inputs", &inputs);
    f(prefix + ".outputs", &outputs);
    f(prefix + ".state", &state);
    f(prefix + ".gi", &gi);
    f(prefix + ".go", &go);
    f(prefix + ".gf", &gf);
    f(prefix + ".ci", &ci);
  }
  Sequence *getState() { return &state; }
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

INetwork *make_SigmoidLayer() { return new SigmoidLayer(); }
INetwork *make_SoftmaxLayer() { return new SoftmaxLayer(); }
INetwork *make_ReluLayer() { return new ReluLayer(); }
INetwork *make_Stacked() { return new Stacked(); }
INetwork *make_Reversed() { return new Reversed(); }
INetwork *make_Parallel() { return new Parallel(); }
INetwork *make_LSTM() { return new NPLSTM(); }
INetwork *make_NPLSTM() { return new NPLSTM(); }

void save_net(const string &file, Network net) {
  save_as_proto(file, net.get());
}
Network load_net(const string &file) { return load_as_proto(file); }

}  // namespace ocropus

#ifdef CLSTM_EXTRAS
// Extra layers; this uses internal function and class definitions from this
// file, so it's included rather than linked. It's mostly a way of slowly
// deprecating
// old layers.
#include "clstm_extras.i"
#endif
