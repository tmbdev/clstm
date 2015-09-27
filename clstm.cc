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

void gradient_clip(Sequence &s, Float m = 1.0) {
  if (m<0) return;
  for (int t = 0; t < s.size(); t++) {
    s[t].d =
        MAPFUNC(s[t].d,
                [m](Float x) { return x > m ? m : x < -m ? -m : x; });
  }
}

void gradient_clip(Mat &d, Float m = 1.0) {
  if (m<0) return;
  d = MAPFUNC(d, [m](Float x) { return x > m ? m : x < -m ? -m : x; });
}

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
  assert(net->outputs.size()==N);
  for (int t = 0; t < N; t++) net->outputs[t].d = targets[t] - net->outputs[t];
}
void set_targets_accelerated(INetwork *net, Sequence &targets) {
  Float lo = 1e-5;
  assert(net->outputs.size() == targets.size());
  int N = net->outputs.size();
  assert(net->outputs.size()==N);
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
  assert(net->outputs.size()==N);
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
  assert(net->outputs.size()==len);
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
  Params W, w;
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
    randinit(w, no, 1, 0.01);
    zeroinit(W.d, no, ni);
    zeroinit(w.d, no, 1);
  }
  void forward() {
    outputs.resize(inputs.size());
    for (int t = 0; t < inputs.size(); t++) {
      outputs[t] = MATMUL(W, inputs[t]);
      Vec v = COL(w, 0);
      ADDCOLS(outputs[t], v);
      NONLIN::f(outputs[t]);
    }
  }
  void backward() {
    for (int t = outputs.size() - 1; t >= 0; t--) {
      NONLIN::df(outputs[t].d, outputs[t]);
      inputs[t].d = MATMUL_TR(W.d, outputs[t].d);
    }
    int bs = COLS(inputs[0]);
    for (int t = 0; t < outputs.size(); t++) {
      W.d += MATMUL_RT(outputs[t].d, inputs[t]);
      for (int b = 0; b < bs; b++) w.d += COL(outputs[t].d, b);
    }
    nseq += 1;
    nsteps += outputs.size();
    outputs[0].d(0, 0) = NAN;  // invalidate it, since we have changed it
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
    W += lr * W.d;
    w += lr * w.d;
    nsteps = 0;
    nseq = 0;
    W.d *= momentum;
    w.d *= momentum;
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
    for (int t = outputs.size() - 1; t >= 0; t--) {
      inputs[t].d = MATMUL_TR(W, outputs[t].d);
    }
    int bs = COLS(inputs[0]);
    for (int t = 0; t < outputs.size(); t++) {
      d_W += MATMUL_RT(outputs[t].d, inputs[t]);
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
    for (int n = sub.size() - 1; n >= 0; n--) {
      if (n + 1 == sub.size())
        for (int t=0; t<outputs.size(); t++)
          sub[n]->outputs[t].d = outputs[t].d;
      else
        for (int t=0; t<sub[n+1]->inputs.size(); t++)
          sub[n]->outputs[t].d = sub[n+1]->inputs[t].d;
      sub[n]->backward();
    }
    for (int t=0; t<sub[0]->inputs.size(); t++)
      inputs[t].d = sub[0]->inputs[t].d;
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

void revcopy(Sequence &out, Sequence &in) {
  revcopy(out.steps, in.steps);
}


struct Reversed : NetworkBase {
  Reversed() { name = "reversed"; }
  const char *kind() { return "Reversed"; }
  int noutput() { return sub[0]->noutput(); }
  int ninput() { return sub[0]->ninput(); }
  void forward() {
    assert(sub.size() == 1);
    INetwork *net = sub[0].get();
    int N = inputs.size();
    net->inputs.resize(N);
    for(int t=0; t<N; t++) net->inputs[t] = inputs[N-t-1];
    net->forward();
    int M = net->outputs.size();
    outputs.resize(M);
    for(int t=0; t<M; t++) outputs[t] = net->outputs[N-t-1];
  }
  void backward() {
    assert(sub.size() == 1);
    INetwork *net = sub[0].get();
    assert(outputs.size() > 0);
    assert(outputs.size() == inputs.size());
    int N = outputs.size();
    assert(net->outputs.size()==outputs.size());
    for(int t=0; t<N; t++) net->outputs[t].d = outputs[N-t-1].d;
    net->backward();
    int M = net->inputs.size();
    assert(inputs.size()==M);
    for(int t=0; t<M; t++) inputs[t].d = net->inputs[N-t-1].d;
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
    int n1 = ROWS(net1->outputs[0]);
    int n2 = ROWS(net2->outputs[0]);
    int N = outputs.size();
    assert(net1->outputs.size() == N);
    assert(net2->outputs.size() == N);
    int bs = COLS(net1->outputs[0]);
    assert(bs == COLS(net2->outputs[0]));
    for (int t = 0; t < N; t++) {
      net1->outputs[t].d.resize(n1, bs);
      net1->outputs[t].d = BLOCK(outputs[t].d, 0, 0, n1, bs);
      net2->outputs[t].d.resize(n2, bs);
      net2->outputs[t].d = BLOCK(outputs[t].d, n1, 0, n2, bs);
    }
    net1->backward();
    net2->backward();
    for (int t = 0; t < N; t++) {
      inputs[t].d = net1->inputs[t].d;
      inputs[t].d += net2->inputs[t].d;
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

#define A array()

// stack the delayed output on the input
void forward_stack1(Batch &all, Batch &inp, Sequence &out, int t) {
  assert(inp.cols() == out.cols());
  int bs = inp.cols();
  int ni = inp.rows();
  int no = out.rows();
  int nf = ni+no+1;
  all.resize(nf, bs);
  BLOCK(all, 0, 0, 1, bs).setConstant(1);
  BLOCK(all, 1, 0, ni, bs) = inp;
  if (t<0)
    BLOCK(all, 1 + ni, 0, no, bs).setConstant(0);
  else
    BLOCK(all, 1 + ni, 0, no, bs) = out[t];
}

void backward_stack1(Batch &all, Batch &inp, Sequence &out, int t) {
  assert(inp.cols() == out.cols());
  int bs = inp.cols();
  int ni = inp.rows();
  int no = out.rows();
  int nf = ni+no+1;
  inp.d += BLOCK(all.d, 1, 0, ni, bs);
  if (t>=0) out[t].d += BLOCK(all.d, 1 + ni, 0, no, bs);
}

// compute non-linear full layers
template <class F>
void forward_full(Batch &y, Params &W, Batch &x) {
  y = nonlin<F>(MATMUL(W, x));
}

template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc) {
  Mat temp = EMUL(yprime<F>(y), y.d);
  gradient_clip(temp, gc);
  x.d += MATMUL_TR(W, temp);
  W.d += MATMUL_RT(temp, x);
}

// combine the delayed gated state with the gated input
void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states, int t, Batch &gf) {
  state = EMUL(ci, gi);
  if (t>=0) state += EMUL(gf, states[t]);
}
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states, int last, Batch &gf) {
  if (last >= 0) states[last].d.A += state.d.A * gf.A;
  if (last >= 0) gf.d.A += state.d.A * states[last].A;
  gi.d.A += state.d.A * ci.A;
  ci.d.A += state.d.A * gi.A;
}

// nonlinear gated output
template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go) {
  out = EMUL(nonlin<H>(state), go);
}
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
  go.d.A += nonlin<H>(state).A * out.d.A;
  state.d.A += xprime<H>(state).A * go.A * out.d.A;
}

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
    outputs.resize(N,no,bs);
    for (int t = 0; t < N; t++) {
      int bs = COLS(inputs[t]);
      forward_stack1(source[t], inputs[t], outputs, t-1);
      forward_full<F>(gi[t], WGI, source[t]);
      forward_full<F>(gf[t], WGF, source[t]);
      forward_full<F>(go[t], WGO, source[t]);
      forward_full<G>(ci[t], WCI, source[t]);
      forward_statemem(state[t], ci[t], gi[t], state, t-1, gf[t]);
      forward_nonlingate<H>(outputs[t], state[t], go[t]);
    }
  }
  void backward() {
    int N = inputs.size();
    int bs = outputs.cols();
    Sequence out;
    out.copy(outputs);
    each([](Sequence &s) {
        s.zeroGrad();
      }, source, inputs, state, gi, go, gf, ci);

    for (int t = N - 1; t >= 0; t--) {
      backward_nonlingate<H>(out[t], state[t], go[t]);
      backward_statemem(state[t], ci[t], gi[t], state, t-1, gf[t]);
      gradient_clip(state[t].d, gradient_clipping);
      backward_full<F>(gi[t], WGI, source[t], gradient_clipping);
      if (t>0) backward_full<F>(gf[t], WGF, source[t], gradient_clipping);
      backward_full<F>(go[t], WGO, source[t], gradient_clipping);
      backward_full<G>(ci[t], WCI, source[t], gradient_clipping);
      backward_stack1(source[t], inputs[t], out, t-1);
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
    each([this,lr](Params &W) {
        W += lr * W.d; W.d *= momentum;
      }, WGI, WGF, WGO, WCI);
  }
  void myweights(const string &prefix, WeightFun f) {
    f(prefix + ".WGI", &WGI, &WGI.d);
    f(prefix + ".WGF", &WGF, &WGF.d);
    f(prefix + ".WGO", &WGO, &WGO.d);
    f(prefix + ".WCI", &WCI, &WCI.d);
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
