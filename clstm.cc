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
  for (int t = 0; t < net->inputs.size(); t++) net->inputs[t] = inputs[t];
}
void set_targets(INetwork *net, Sequence &targets) {
  int N = net->outputs.size();
  assert(N == targets.size());
  net->d_outputs.resize(N);
  for (int t = 0; t < N; t++) net->d_outputs[t] = targets[t] - net->outputs[t];
}
void set_targets_accelerated(INetwork *net, Sequence &targets) {
  Float lo = 1e-5;
  assert(net->outputs.size() == targets.size());
  int N = net->outputs.size();
  net->d_outputs.resize(N);
  for (int t = 0; t < N; t++) {
    net->d_outputs[t] = -net->outputs[t];
    for (int i = 0; i < ROWS(targets[t]); i++) {
      for (int b = 0; b < COLS(targets[t]); b++) {
        // only allow binary classification
        assert(fabs(targets[t](i, b) - 0) < 1e-5 ||
               fabs(targets[t](i, b) - 1) < 1e-5);
        if (targets[t](i, b) > 0.5) {
          net->d_outputs[t](i, b) = 1.0 / fmax(lo, net->outputs[t](i, b));
        }
      }
    }
  }
}
void set_classes(INetwork *net, Classes &classes) {
  int N = net->outputs.size();
  assert(N == classes.size());
  net->d_outputs.resize(N);
  for (int t = 0; t < N; t++) {
    net->d_outputs[t] = -net->outputs[t];
    net->d_outputs[t](classes[t]) += 1;
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
  net->d_outputs.resize(len);
  if (dim == 1) {
    for (int t = 0; t < len; t++)
      net->d_outputs[t](0) =
          cs[t] ? 1.0 - net->outputs[t](0) : -net->outputs[t](0);
  } else {
    for (int t = 0; t < len; t++) {
      net->d_outputs[t] = -net->outputs[t];
      int c = cs[t];
      net->d_outputs[t](c) = 1 - net->outputs[t](c);
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
  net->d_outputs.resize(len);
  if (dim == 1) {
    for (int t = 0; t < len; t++) {
      if (cs[t] == 0)
        net->d_outputs[t](0) = -1.0 / fmax(lo, 1.0 - net->outputs[t](0));
      else
        net->d_outputs[t](0) = 1.0 / fmax(lo, net->outputs[t](0));
    }
  } else {
    for (int t = 0; t < len; t++) {
      net->d_outputs[t] = -net->outputs[t];
      int c = cs[t];
      net->d_outputs[t](c) = 1.0 / fmax(lo, net->outputs[t](c));
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

void INetwork::states(const string &prefix, StateFun f) {
  string nprefix = prefix + "." + name;
  f(nprefix + ".inputs", &inputs);
  f(nprefix + ".d_inputs", &d_inputs);
  f(nprefix + ".outputs", &outputs);
  f(nprefix + ".d_outputs", &d_outputs);
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
    d_outputs.resize(outputs.size());
    for (int t = 0; t < outputs.size(); t++) {
      Vec delta = targets[t] - outputs[t];
      total += delta.array().square().sum();
      d_outputs[t] = delta;
    }
    backward();
    update();
    return total;
  }
};

inline Float limexp(Float x) {
#if 1
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
#else
  return exp(x);
#endif
}

inline Float sigmoid(Float x) {
#if 1
  return 1.0 / (1.0 + limexp(-x));
#else
  return 1.0 / (1.0 + exp(-x));
#endif
}

template <class NONLIN>
struct Full : NetworkBase {
  Mat W, d_W;
  Vec w, d_w;
  int nseq = 0;
  int nsteps = 0;
  string mykind = string("Full_") + NONLIN::kind;
  Full() { name = string("full_") + NONLIN::name; }
  const char *kind() { return mykind.c_str(); }
  int noutput() { return ROWS(W); }
  int ninput() { return COLS(W); }
  void initialize() {
    int no = irequire("noutput");
    int ni = irequire("ninput");
    randinit(W, no, ni, 0.01);
    randinit(w, no, 0.01);
    zeroinit(d_W, no, ni);
    zeroinit(d_w, no);
  }
  void forward() {
    outputs.resize(inputs.size());
    for (int t = 0; t < inputs.size(); t++) {
      outputs[t] = MATMUL(W, inputs[t]);
      ADDCOLS(outputs[t], w);
      NONLIN::f(outputs[t]);
    }
  }
  void backward() {
    d_inputs.resize(d_outputs.size());
    for (int t = d_outputs.size() - 1; t >= 0; t--) {
      NONLIN::df(d_outputs[t], outputs[t]);
      d_inputs[t] = MATMUL_TR(W, d_outputs[t]);
    }
    int bs = COLS(inputs[0]);
    for (int t = 0; t < d_outputs.size(); t++) {
      d_W += MATMUL_RT(d_outputs[t], inputs[t]);
      for (int b = 0; b < bs; b++) d_w += COL(d_outputs[t], b);
    }
    nseq += 1;
    nsteps += d_outputs.size();
    d_outputs[0](0, 0) = NAN;  // invalidate it, since we have changed it
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
    W += lr * d_W;
    w += lr * d_w;
    nsteps = 0;
    nseq = 0;
    d_W *= momentum;
    d_w *= momentum;
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".W", &W, (Mat *)0);
    f(prefix + ".w", &w, (Vec *)0);
  }
};

struct NoNonlin {
  static constexpr const char *kind = "Linear";
  static constexpr const char *name = "linear";
  template <class T>
  static void f(T &x) {}
  template <class T, class U>
  static void df(T &dx, U &y) {}
};
typedef Full<NoNonlin> LinearLayer;
REGISTER(LinearLayer);

struct SigmoidNonlin {
  static constexpr const char *kind = "Sigmoid";
  static constexpr const char *name = "sigmoid";
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, sigmoid);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= y.array() * (1 - y.array());
  }
};
typedef Full<SigmoidNonlin> SigmoidLayer;
REGISTER(SigmoidLayer);

Float tanh_(Float x) { return tanh(x); }
struct TanhNonlin {
  static constexpr const char *kind = "Tanh";
  static constexpr const char *name = "tanh";
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, tanh_);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= (1 - y.array().square());
  }
};
typedef Full<TanhNonlin> TanhLayer;
REGISTER(TanhLayer);

inline Float relu_(Float x) { return x <= 0 ? 0 : x; }
inline Float heavi_(Float x) { return x <= 0 ? 0 : 1; }
struct ReluNonlin {
  static constexpr const char *kind = "Relu";
  static constexpr const char *name = "relu";
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, relu_);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= MAPFUN(y, heavi_).array();
  }
};
typedef Full<ReluNonlin> ReluLayer;
REGISTER(ReluLayer);

struct SoftmaxLayer : NetworkBase {
  Mat W, d_W;
  Vec w, d_w;
  int nsteps = 0;
  int nseq = 0;
  SoftmaxLayer() { name = "softmax"; }
  const char *kind() { return "SoftmaxLayer"; }
  int noutput() { return ROWS(W); }
  int ninput() { return COLS(W); }
  void initialize() {
    int no = irequire("noutput");
    int ni = irequire("ninput");
    if (no < 2) THROW("Softmax requires no>=2");
    randinit(W, no, ni, 0.01);
    randinit(w, no, 0.01);
    clearUpdates();
  }
  void clearUpdates() {
    int no = ROWS(W);
    int ni = COLS(W);
    zeroinit(d_W, no, ni);
    zeroinit(d_w, no);
  }
  void postLoad() {
    clearUpdates();
    makeEncoders();
  }
  void forward() {
    outputs.resize(inputs.size());
    int no = ROWS(W), bs = COLS(inputs[0]);
    for (int t = 0; t < inputs.size(); t++) {
      outputs[t].resize(no, bs);
      for (int b = 0; b < COLS(outputs[t]); b++) {
        COL(outputs[t], b) = MAPFUN(DOT(W, COL(inputs[t], b)) + w, limexp);
        Float total = fmax(SUMREDUCE(COL(outputs[t], b)), 1e-9);
        COL(outputs[t], b) /= total;
      }
    }
  }
  void backward() {
    d_inputs.resize(d_outputs.size());
    for (int t = d_outputs.size() - 1; t >= 0; t--) {
      d_inputs[t] = MATMUL_TR(W, d_outputs[t]);
    }
    int bs = COLS(inputs[0]);
    for (int t = 0; t < d_outputs.size(); t++) {
      d_W += MATMUL_RT(d_outputs[t], inputs[t]);
      for (int b = 0; b < bs; b++) d_w += COL(d_outputs[t], b);
    }
    nsteps += d_outputs.size();
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
    W += lr * d_W;
    w += lr * d_w;
    nsteps = 0;
    nseq = 0;
    d_W *= momentum;
    d_w *= momentum;
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".W", &W, &d_W);
    f(prefix + ".w", &w, &d_w);
  }
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
    assert(d_outputs.size() > 0);
    assert(d_outputs.size() == outputs.size());
    for (int n = sub.size() - 1; n >= 0; n--) {
      if (n + 1 == sub.size())
        sub[n]->d_outputs = d_outputs;
      else
        sub[n]->d_outputs = sub[n + 1]->d_inputs;
      sub[n]->backward();
    }
    d_inputs = sub[0]->d_inputs;
  }
  void update() {
    for (int i = 0; i < sub.size(); i++) sub[i]->update();
  }
};
REGISTER(Stacked);

template <class T>
inline void revcopy(vector<T> &out, vector<T> &in) {
  int N = in.size();
  out.resize(N);
  for (int i = 0; i < N; i++) out[i] = in[N - i - 1];
}

struct Reversed : NetworkBase {
  Reversed() { name = "reversed"; }
  const char *kind() { return "Reversed"; }
  int noutput() { return sub[0]->noutput(); }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(sub.size() == 1);
    INetwork *net = sub[0].get();
    revcopy(net->inputs, inputs);
    net->forward();
    revcopy(outputs, net->outputs);
  }
  void backward() {
    assert(sub.size() == 1);
    INetwork *net = sub[0].get();
    assert(outputs.size() > 0);
    assert(outputs.size() == inputs.size());
    assert(d_outputs.size() > 0);
    revcopy(net->d_outputs, d_outputs);
    net->backward();
    revcopy(d_inputs, net->d_inputs);
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
    INetwork *net1 = sub[0].get();
    INetwork *net2 = sub[1].get();
    net1->inputs = inputs;
    net2->inputs = inputs;
    net1->forward();
    net2->forward();
    int N = inputs.size();
    assert(net1->outputs.size() == N);
    assert(net2->outputs.size() == N);
    int n1 = ROWS(net1->outputs[0]);
    int n2 = ROWS(net2->outputs[0]);
    outputs.resize(N);
    int bs = COLS(net1->outputs[0]);
    assert(bs == COLS(net2->outputs[0]));
    for (int t = 0; t < N; t++) {
      outputs[t].resize(n1 + n2, bs);
      BLOCK(outputs[t], 0, 0, n1, bs) = net1->outputs[t];
      BLOCK(outputs[t], n1, 0, n2, bs) = net2->outputs[t];
    }
  }
  void backward() {
    assert(sub.size() == 2);
    INetwork *net1 = sub[0].get();
    INetwork *net2 = sub[1].get();
    assert(outputs.size() > 0);
    assert(outputs.size() == inputs.size());
    assert(d_outputs.size() > 0);
    int n1 = ROWS(net1->outputs[0]);
    int n2 = ROWS(net2->outputs[0]);
    int N = outputs.size();
    net1->d_outputs.resize(N);
    net2->d_outputs.resize(N);
    int bs = COLS(net1->outputs[0]);
    assert(bs == COLS(net2->outputs[0]));
    for (int t = 0; t < N; t++) {
      net1->d_outputs[t].resize(n1, bs);
      net1->d_outputs[t] = BLOCK(d_outputs[t], 0, 0, n1, bs);
      net2->d_outputs[t].resize(n2, bs);
      net2->d_outputs[t] = BLOCK(d_outputs[t], n1, 0, n2, bs);
    }
    net1->backward();
    net2->backward();
    d_inputs.resize(N);
    for (int t = 0; t < N; t++) {
      d_inputs[t] = net1->d_inputs[t];
      d_inputs[t] += net2->d_inputs[t];
    }
  }
  void update() {
    for (int i = 0; i < sub.size(); i++) sub[i]->update();
  }
};
REGISTER(Parallel);

namespace {
template <class NONLIN, class T>
inline Mat nonlin(T &a) {
  Mat result = a;
  NONLIN::f(result);
  return result;
}
template <class NONLIN, class T>
inline Mat yprime(T &a) {
  Mat result = Mat::Ones(ROWS(a), COLS(a));
  NONLIN::df(result, a);
  return result;
}
template <class NONLIN, class T>
inline Mat xprime(T &a) {
  Mat result = Mat::Ones(ROWS(a), COLS(a));
  Mat temp = a;
  NONLIN::f(temp);
  NONLIN::df(result, temp);
  return result;
}
template <typename F, typename T>
void each(F f, T &a) {
  f(a);
}
template <typename F, typename T, typename... Args>
void each(F f, T &a, Args &&... args) {
  f(a);
  each(f, args...);
}
}

template <class F = SigmoidNonlin, class G = TanhNonlin, class H = TanhNonlin>
struct GenericNPLSTM : NetworkBase {
#define SEQUENCES gi, gf, go, ci, state
#define DSEQUENCES gierr, gferr, goerr, cierr, stateerr, outerr
#define WEIGHTS WGI, WGF, WGO, WCI
#define DWEIGHTS DWGI, DWGF, DWGO, DWCI
  Sequence source, SEQUENCES, sourceerr, DSEQUENCES;
  Mat WEIGHTS, DWEIGHTS;
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
    each([this](Mat &d) { d = Mat::Zero(no, nf); }, DWEIGHTS);
  }
  void resize(int N) {
    each([N](Sequence &s) {
      s.resize(N);
      for (int t = 0; t < N; t++) s[t].setConstant(NAN);
    }, source, sourceerr, outputs, SEQUENCES, DSEQUENCES);
    assert(source.size() == N);
    assert(gi.size() == N);
    assert(goerr.size() == N);
  }
#define A array()
  void forward() {
    int N = inputs.size();
    resize(N);
    for (int t = 0; t < N; t++) {
      int bs = COLS(inputs[t]);
      source[t].resize(nf, bs);
      BLOCK(source[t], 0, 0, 1, bs).setConstant(1);
      BLOCK(source[t], 1, 0, ni, bs) = inputs[t];
      if (t == 0)
        BLOCK(source[t], 1 + ni, 0, no, bs).setConstant(0);
      else
        BLOCK(source[t], 1 + ni, 0, no, bs) = outputs[t - 1];
      gi[t] = nonlin<F>(MATMUL(WGI, source[t]));
      gf[t] = nonlin<F>(MATMUL(WGF, source[t]));
      go[t] = nonlin<F>(MATMUL(WGO, source[t]));
      ci[t] = nonlin<G>(MATMUL(WCI, source[t]));
      state[t] = ci[t].A * gi[t].A;
      if (t > 0) state[t] += EMUL(gf[t], state[t - 1]);
      outputs[t] = nonlin<H>(state[t]).A * go[t].A;
    }
  }
  void backward() {
    int N = inputs.size();
    d_inputs.resize(N);
    for (int t = N - 1; t >= 0; t--) {
      int bs = COLS(d_outputs[t]);
      outerr[t] = d_outputs[t];
      if (t < N - 1) outerr[t] += BLOCK(sourceerr[t + 1], 1 + ni, 0, no, bs);
      goerr[t] = EMUL(EMUL(yprime<F>(go[t]), nonlin<H>(state[t])), outerr[t]);
      stateerr[t] = EMUL(EMUL(xprime<H>(state[t]), go[t].A), outerr[t]);
      if (t < N - 1) stateerr[t] += EMUL(stateerr[t + 1], gf[t + 1]);
      if (t > 0)
        gferr[t] = EMUL(EMUL(yprime<F>(gf[t]), stateerr[t]), state[t - 1]);
      gierr[t] = EMUL(EMUL(yprime<F>(gi[t]), stateerr[t]), ci[t]);
      cierr[t] = EMUL(EMUL(yprime<G>(ci[t]), stateerr[t]), gi[t]);
      sourceerr[t] = MATMUL_TR(WGI, gierr[t]);
      if (t > 0) sourceerr[t] += MATMUL_TR(WGF, gferr[t]);
      sourceerr[t] += MATMUL_TR(WGO, goerr[t]);
      sourceerr[t] += MATMUL_TR(WCI, cierr[t]);
      d_inputs[t].resize(ni, bs);
      d_inputs[t] = BLOCK(sourceerr[t], 1, 0, ni, bs);
    }
    if (gradient_clipping > 0 || gradient_clipping < 999) {
      gradient_clip(gierr, gradient_clipping);
      gradient_clip(gferr, gradient_clipping);
      gradient_clip(goerr, gradient_clipping);
      gradient_clip(cierr, gradient_clipping);
    }
    for (int t = 0; t < N; t++) {
      DWGI += MATMUL_RT(gierr[t], source[t]);
      if (t > 0) DWGF += MATMUL_RT(gferr[t], source[t]);
      DWGO += MATMUL_RT(goerr[t], source[t]);
      DWCI += MATMUL_RT(cierr[t], source[t]);
    }
    nsteps += N;
    nseq += 1;
  }
#undef A
  void gradient_clip(Sequence &s, Float m = 1.0) {
    for (int t = 0; t < s.size(); t++) {
      s[t] =
          MAPFUNC(s[t], [m](Float x) { return x > m ? m : x < -m ? -m : x; });
    }
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
    WGI += lr * DWGI;
    WGF += lr * DWGF;
    WGO += lr * DWGO;
    WCI += lr * DWCI;
    DWGI *= momentum;
    DWGF *= momentum;
    DWGO *= momentum;
    DWCI *= momentum;
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".WGI", &WGI, &DWGI);
    f(prefix + ".WGF", &WGF, &DWGF);
    f(prefix + ".WGO", &WGO, &DWGO);
    f(prefix + ".WCI", &WCI, &DWCI);
  }
  virtual void mystates(const string &prefix, StateFun f) {
    f(prefix + ".inputs", &inputs);
    f(prefix + ".d_inputs", &d_inputs);
    f(prefix + ".outputs", &outputs);
    f(prefix + ".d_outputs", &d_outputs);
    f(prefix + ".state", &state);
    f(prefix + ".stateerr", &stateerr);
    f(prefix + ".gi", &gi);
    f(prefix + ".gierr", &gierr);
    f(prefix + ".go", &go);
    f(prefix + ".goerr", &goerr);
    f(prefix + ".gf", &gf);
    f(prefix + ".gferr", &gferr);
    f(prefix + ".ci", &ci);
    f(prefix + ".cierr", &cierr);
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
Network load_net(const string &file) {
  return load_as_proto(file);
}

}  // namespace ocropus

#ifdef CLSTM_EXTRAS
// Extra layers; this uses internal function and class definitions from this
// file, so it's included rather than linked. It's mostly a way of slowly
// deprecating
// old layers.
#include "clstm_extras.i"
#endif
