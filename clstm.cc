#include "clstm.h"
#include "h5eigen.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {
char exception_message[256];
map<string, INetworkFactory> network_factories;

template <class T>
int register_network(const char *name) {
    string s(name);
    network_factories[s] = [] () { return new T(); };
    return 0;
}
#define C(X, Y) X ## Y
#define REGISTER(X) int C(status_, X) = register_network<X>(# X);

Mat debugmat;

using namespace std;
using Eigen::Ref;

bool no_update = false;
bool verbose = false;

void set_inputs(INetwork *net, Sequence &inputs) {
    net->inputs.resize(inputs.size());
    for (int t = 0; t < net->inputs.size(); t++)
        net->inputs[t] = inputs[t];
}
void set_targets(INetwork *net, Sequence &targets) {
    int N = net->outputs.size();
    assert(N == targets.size());
    net->d_outputs.resize(N);
    for (int t = 0; t < N; t++)
        net->d_outputs[t] = targets[t] - net->outputs[t];
}
void set_targets_accelerated(INetwork *net, Sequence &targets) {
    Float lo = 1e-5;
    assert(net->outputs.size() == targets.size());
    int N = net->outputs.size();
    net->d_outputs.resize(N);
    for (int t = 0; t < N; t++) {
        net->d_outputs[t] = -net->outputs[t];
        for (int i = 0; i < targets[t].rows(); i++) {
            for (int b = 0; b < targets[t].cols(); b++) {
                // only allow binary classification
                assert(fabs(targets[t](i, b)-0) < 1e-5 || fabs(targets[t](i, b)-1) < 1e-5);
                if (targets[t](i, b) > 0.5) {
                    net->d_outputs[t](i, b) = 1.0/fmax(lo, net->outputs[t](i, b));
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
            net->d_outputs[t](0) = cs[t] ?
                1.0-net->outputs[t](0) : -net->outputs[t](0);
    } else {
        for (int t = 0; t < len; t++) {
            net->d_outputs[t] = -net->outputs[t];
            int c = cs[t];
            net->d_outputs[t](c) = 1-net->outputs[t](c);
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
                net->d_outputs[t](0) = -1.0/fmax(lo, 1.0-net->outputs[t](0));
            else
                net->d_outputs[t](0) = 1.0/fmax(lo, net->outputs[t](0));
        }
    } else {
        for (int t = 0; t < len; t++) {
            net->d_outputs[t] = -net->outputs[t];
            int c = cs[t];
            net->d_outputs[t](c) = 1.0/fmax(lo, net->outputs[t](c));
        }
    }
    net->backward();
    net->update();
}

void cpred(INetwork *net, Classes &preds, Sequence &xs) {
    int N = xs.size();
    assert(xs[0].cols() == 0);
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

void INetwork::encode(Classes &classes, std::wstring &s) {
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
void INetwork::iencode(Classes &classes, std::wstring &s) {
    if (!iencoder) makeEncoders();
    classes.clear();
    for (int pos = 0; pos < s.size(); pos++) {
        int c = (*iencoder)[int(s[pos])];
        classes.push_back(c);
    }
}
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

void INetwork::info(string prefix){
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
        sub[i]->weights(nprefix+"."+to_string(i), f);
    }
}

void INetwork::states(const string &prefix, StateFun f) {
    string nprefix = prefix + "." + name;
    f(nprefix+".inputs", &inputs);
    f(nprefix+".d_inputs", &d_inputs);
    f(nprefix+".outputs", &outputs);
    f(nprefix+".d_outputs", &d_outputs);
    mystates(nprefix, f);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->states(nprefix+"."+to_string(i), f);
    }
}

void INetwork::networks(const string &prefix, function<void (string, INetwork*)> f) {
    string nprefix = prefix+"."+name;
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

struct Network : INetwork {
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
struct Full : Network {
    Mat W, d_W;
    Vec w, d_w;
    int nseq = 0;
    int nsteps = 0;
    Full() {
        name = "full_";
        name += NONLIN::kind;
    }
    int noutput() {
        return W.rows();
    }
    int ninput() {
        return W.cols();
    }
    void initialize() {
        int no = irequire("noutput");
        int ni = irequire("ninput");
        W = Mat::Random(no, ni) * 0.01;
        w = Vec::Random(no) * 0.01;
        d_W = Mat::Zero(no, ni);
        d_w = Vec::Zero(no);
    }
    void forward() {
        outputs.resize(inputs.size());
        for (int t = 0; t < inputs.size(); t++) {
            int bs = outputs[t].cols();
            outputs[t] = W * inputs[t];
            for (int b = 0; b < bs; b++) outputs[t].col(b) += w;
            NONLIN::f(outputs[t]);
        }
    }
    void backward() {
        d_inputs.resize(d_outputs.size());
        for (int t = d_outputs.size()-1; t >= 0; t--) {
            NONLIN::df(d_outputs[t], outputs[t]);
            d_inputs[t] = W.transpose() * d_outputs[t];
        }
        for (int t = 0; t < d_outputs.size(); t++) {
            int bs = outputs[t].cols();
            d_W += d_outputs[t] * inputs[t].transpose();
            for (int b = 0; b < bs; b++) d_w += d_outputs[t].col(b);
        }
        nseq += 1;
        nsteps += d_outputs.size();
        d_outputs[0](0, 0) = NAN;
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else throw "unknown normalization";
        W += lr * d_W;
        w += lr * d_w;
        nsteps = 0;
        nseq = 0;
        d_W *= momentum;
        d_w *= momentum;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".W", &W, (Mat*)0);
        f(prefix+".w", &w, (Vec*)0);
    }
};

struct NoNonlin {
    static constexpr const char *kind = "linear";
    template <class T>
    static void f(T &x) {
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
    }
};
typedef Full<NoNonlin> LinearLayer;
REGISTER(LinearLayer);

struct SigmoidNonlin {
    static constexpr const char *kind = "sigmoid";
    template <class T>
    static void f(T &x) {
        x = x.unaryExpr(ptr_fun(sigmoid));
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= y.array() * (1-y.array());
    }
};
typedef Full<SigmoidNonlin> SigmoidLayer;
REGISTER(SigmoidLayer);

Float tanh_(Float x) {
    return tanh(x);
}
struct TanhNonlin {
    static constexpr const char *kind = "tanh";
    template <class T>
    static void f(T &x) {
        x = x.unaryExpr(ptr_fun(tanh_));
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= (1 - y.array().square());
    }
};
typedef Full<TanhNonlin> TanhLayer;
REGISTER(TanhLayer);

inline Float relu_(Float x) {
    return x <= 0 ? 0 : x;
}
inline Float heavi_(Float x) {
    return x <= 0 ? 0 : 1;
}
struct ReluNonlin {
    static constexpr const char *kind = "relu";
    template <class T>
    static void f(T &x) {
        x = x.unaryExpr(ptr_fun(relu_));
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= y.unaryExpr(ptr_fun(heavi_)).array();
    }
};
typedef Full<ReluNonlin> ReluLayer;
REGISTER(ReluLayer);

INetwork *make_LinearLayer() {
    return new LinearLayer();
}

INetwork *make_SigmoidLayer() {
    return new SigmoidLayer();
}

INetwork *make_TanhLayer() {
    return new TanhLayer();
}

INetwork *make_ReluLayer() {
    return new TanhLayer();
}

struct SoftmaxLayer : Network {
    Mat W, d_W;
    Vec w, d_w;
    int nsteps = 0;
    int nseq = 0;
    SoftmaxLayer() {
        name = "softmax";
    }
    int noutput() {
        return W.rows();
    }
    int ninput() {
        return W.cols();
    }
    void initialize() {
        int no = irequire("noutput");
        int ni = irequire("ninput");
        if (no < 2) throw "Softmax requires no>=2";
        W = Mat::Random(no, ni) * 0.01;
        w = Vec::Random(no) * 0.01;
        clearUpdates();
    }
    void clearUpdates() {
        int no = W.rows();
        int ni = W.cols();
        d_W = Mat::Zero(no, ni);
        d_w = Vec::Zero(no);
    }
    void postLoad() {
        clearUpdates();
        makeEncoders();
    }
    void forward() {
        outputs.resize(inputs.size());
        int no = W.rows(), bs = inputs[0].cols();
        for (int t = 0; t < inputs.size(); t++) {
            outputs[t].resize(no, bs);
            for (int b = 0; b < outputs[t].cols(); b++) {
                outputs[t].col(b) = (W * inputs[t].col(b) + w).array().unaryExpr(ptr_fun(limexp));
                Float total = fmax(outputs[t].col(b).sum(), 1e-9);
                outputs[t].col(b) /= total;
            }
        }
    }
    void backward() {
        d_inputs.resize(d_outputs.size());
        for (int t = d_outputs.size()-1; t >= 0; t--) {
            d_inputs[t] = W.transpose() * d_outputs[t];
        }
        for (int t = 0; t < d_outputs.size(); t++) {
            d_W += d_outputs[t] * inputs[t].transpose();
            int bs = d_outputs[t].cols();
            for (int b = 0; b < bs; b++) d_w += d_outputs[t].col(b);
        }
        nsteps += d_outputs.size();
        nseq += 1;
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else throw "unknown normalization";
        W += lr * d_W;
        w += lr * d_w;
        nsteps = 0;
        nseq = 0;
        d_W *= momentum;
        d_w *= momentum;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".W", &W, &d_W);
        f(prefix+".w", &w, &d_w);
    }
};
REGISTER(SoftmaxLayer);

INetwork *make_SoftmaxLayer() {
    return new SoftmaxLayer();
}

struct Stacked : Network {
    Stacked() {
        name = "stacked";
    }
    int noutput() {
        return sub[sub.size()-1]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        assert(inputs.size() > 0);
        assert(sub.size() > 0);
        for (int n = 0; n < sub.size(); n++) {
            if (n == 0) sub[n]->inputs = inputs;
            else sub[n]->inputs = sub[n-1]->outputs;
            sub[n]->forward();
        }
        outputs = sub[sub.size()-1]->outputs;
        assert(outputs.size() == inputs.size());
    }
    void backward() {
        assert(outputs.size() > 0);
        assert(outputs.size() == inputs.size());
        assert(d_outputs.size() > 0);
        assert(d_outputs.size() == outputs.size());
        for (int n = sub.size()-1; n >= 0; n--) {
            if (n+1 == sub.size()) sub[n]->d_outputs = d_outputs;
            else sub[n]->d_outputs = sub[n+1]->d_inputs;
            sub[n]->backward();
        }
        d_inputs = sub[0]->d_inputs;
    }
    void update() {
        for (int i = 0; i < sub.size(); i++)
            sub[i]->update();
    }
};
REGISTER(Stacked);

INetwork *make_Stacked() {
    return new Stacked();
}

template <class T>
inline void revcopy(vector<T> &out, vector<T> &in) {
    int N = in.size();
    out.resize(N);
    for (int i = 0; i < N; i++) out[i] = in[N-i-1];
}

struct Reversed : Network {
    Reversed() {
        name = "reversed";
    }
    int noutput() {
        return sub[0]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
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
    void update() {
        sub[0]->update();
    }
};
REGISTER(Reversed);

INetwork *make_Reversed() {
    return new Reversed();
}

struct Parallel : Network {
    Parallel() {
        name = "parallel";
    }
    int noutput() {
        return sub[0]->noutput() + sub[1]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
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
        int n1 = net1->outputs[0].rows();
        int n2 = net2->outputs[0].rows();
        outputs.resize(N);
        int bs = net1->outputs[0].cols();
        assert(bs == net2->outputs[0].cols());
        for (int t = 0; t < N; t++) {
            outputs[t].resize(n1+n2, bs);
            outputs[t].block(0, 0, n1, bs) = net1->outputs[t];
            outputs[t].block(n1, 0, n2, bs) = net2->outputs[t];
        }
    }
    void backward() {
        assert(sub.size() == 2);
        INetwork *net1 = sub[0].get();
        INetwork *net2 = sub[1].get();
        assert(outputs.size() > 0);
        assert(outputs.size() == inputs.size());
        assert(d_outputs.size() > 0);
        int n1 = net1->outputs[0].rows();
        int n2 = net2->outputs[0].rows();
        int N = outputs.size();
        net1->d_outputs.resize(N);
        net2->d_outputs.resize(N);
        int bs = net1->outputs[0].cols();
        assert(bs == net2->outputs[0].cols());
        for (int t = 0; t < N; t++) {
            net1->d_outputs[t].resize(n1, bs);
            net1->d_outputs[t] = d_outputs[t].block(0, 0, n1, bs);
            net2->d_outputs[t].resize(n2, bs);
            net2->d_outputs[t] = d_outputs[t].block(n1, 0, n2, bs);
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

INetwork *make_Parallel() {
    return new Parallel();
}

namespace {
template <class NONLIN, class T>
inline Mat nonlin(T &a) {
    Mat result = a;
    NONLIN::f(result);
    return result;
}
template <class NONLIN, class T>
inline Mat yprime(T &a) {
    Mat result = Mat::Ones(a.rows(), a.cols());
    NONLIN::df(result, a);
    return result;
}
template <class NONLIN, class T>
inline Mat xprime(T &a) {
    Mat result = Mat::Ones(a.rows(), a.cols());
    Mat temp = a;
    NONLIN::f(temp);
    NONLIN::df(result, temp);
    return result;
}
template <typename F, typename T>
void each(F f, T &a) {
    f(a);
}
template <typename F, typename T, typename ... Args>
void each(F f, T &a, Args&&... args) {
    f(a);
    each(f, args ...);
}
}

template <class F, class G, class H, bool PEEP = true>
struct GenericLSTM : Network {
    // NB: verified gradients against Python implementation; this
    // code yields identical numerical results
#define SEQUENCES gix, gfx, gox, cix, gi, gf, go, ci, state
#define DSEQUENCES gierr, gferr, goerr, cierr, stateerr, outerr
#define WEIGHTS WGI, WGF, WGO, WCI
#define PEEPS WIP, WFP, WOP
#define DWEIGHTS DWGI, DWGF, DWGO, DWCI
#define DPEEPS DWIP, DWFP, DWOP
    Sequence source, SEQUENCES, sourceerr, DSEQUENCES;
    Mat WEIGHTS, DWEIGHTS;
    Vec PEEPS, DPEEPS;
    Float gradient_clipping = 10.0;
    int ni, no, nf;
    int nsteps = 0;
    int nseq = 0;
    GenericLSTM() {
        name = "lstm";
    }
    int noutput() {
        return no;
    }
    int ninput() {
        return ni;
    }
    void postLoad() {
        no = WGI.rows();
        nf = WGI.cols();
        assert(nf > no);
        ni = nf-no-1;
        clearUpdates();
    }
    void initialize() {
        int ni = irequire("ninput");
        int no = irequire("noutput");
        int nf = 1+ni+no;
        this->ni = ni;
        this->no = no;
        this->nf = nf;
        each([no, nf](Mat &w) {
                 w = Mat::Random(no, nf) * 0.01;
             }, WEIGHTS);
        if (PEEP) {
            each([no](Vec &w) {
                     w = Vec::Random(no) * 0.01;
                 }, PEEPS);
        };
        clearUpdates();
    }
    void clearUpdates() {
        each([this](Mat &d) { d = Mat::Zero(no, nf); }, DWEIGHTS);
        if (PEEP) each([this](Vec &d) { d = Vec::Zero(no); }, DPEEPS);
    }
    void resize(int N) {
        each([N](Sequence &s) {
                 s.resize(N);
                 for (int t = 0; t < N; t++) s[t].setConstant(NAN);
             }, source, sourceerr, outputs, SEQUENCES, DSEQUENCES);
        assert(source.size() == N);
        assert(gix.size() == N);
        assert(goerr.size() == N);
    }
#define A array()
    void forward() {
        int N = inputs.size();
        resize(N);
        for (int t = 0; t < N; t++) {
            int bs = inputs[t].cols();
            source[t].resize(nf, bs);
            source[t].block(0, 0, 1, bs).setConstant(1);
            source[t].block(1, 0, ni, bs) = inputs[t];
            if (t == 0) source[t].block(1+ni, 0, no, bs).setConstant(0);
            else source[t].block(1+ni, 0, no, bs) = outputs[t-1];
            gix[t] = WGI * source[t];
            gfx[t] = WGF * source[t];
            gox[t] = WGO * source[t];
            cix[t] = WCI * source[t];
            if (t > 0) {
                int bs = state[t-1].cols();
                for (int b = 0; b < bs; b++) {
                    if (PEEP) gix[t].col(b).A += WIP.A * state[t-1].col(b).A;
                    if (PEEP) gfx[t].col(b).A += WFP.A * state[t-1].col(b).A;
                }
            }
            gi[t] = nonlin<F>(gix[t]);
            gf[t] = nonlin<F>(gfx[t]);
            ci[t] = nonlin<G>(cix[t]);
            state[t] = ci[t].A * gi[t].A;
            if (t > 0) {
                state[t].A += gf[t].A * state[t-1].A;
                if (PEEP) {
                    int bs = state[t].cols();
                    for (int b = 0; b < bs; b++)
                        gox[t].col(b).A += WOP.A * state[t].col(b).A;
                }
            }
            go[t] = nonlin<F>(gox[t]);
            outputs[t] = nonlin<H>(state[t]).A * go[t].A;
        }
    }
    void backward() {
        int N = inputs.size();
        d_inputs.resize(N);
        for (int t = N-1; t >= 0; t--) {
            int bs = d_outputs[t].cols();
            outerr[t] = d_outputs[t];
            if (t < N-1) outerr[t] += sourceerr[t+1].block(1+ni, 0, no, bs);
            goerr[t] = yprime<F>(go[t]).A * nonlin<H>(state[t]).A * outerr[t].A;
            stateerr[t] = xprime<H>(state[t]).A * go[t].A * outerr[t].A;
            if (PEEP) {
                for (int b = 0; b < bs; b++)
                    stateerr[t].col(b).A += goerr[t].col(b).A * WOP.A;
            }
            if (t < N-1) {
                if (PEEP) for (int b = 0; b < bs; b++) {
                        stateerr[t].col(b).A += gferr[t+1].col(b).A * WFP.A;
                        stateerr[t].col(b).A += gierr[t+1].col(b).A * WIP.A;
                    }
                stateerr[t].A += stateerr[t+1].A * gf[t+1].A;
            }
            if (t > 0) gferr[t] = yprime<F>(gf[t]).A * stateerr[t].A * state[t-1].A;
            gierr[t] = yprime<F>(gi[t]).A * stateerr[t].A * ci[t].A;
            cierr[t] = yprime<G>(ci[t]).A * stateerr[t].A * gi[t].A;
            sourceerr[t] = WGI.transpose() * gierr[t];
            if (t > 0) sourceerr[t] += WGF.transpose() * gferr[t];
            sourceerr[t] += WGO.transpose() * goerr[t];
            sourceerr[t] += WCI.transpose() * cierr[t];
            d_inputs[t].resize(ni, bs);
            d_inputs[t] = sourceerr[t].block(1, 0, ni, bs);
        }
        if (gradient_clipping > 0 || gradient_clipping < 999) {
            gradient_clip(gierr, gradient_clipping);
            gradient_clip(gferr, gradient_clipping);
            gradient_clip(goerr, gradient_clipping);
            gradient_clip(cierr, gradient_clipping);
        }
        for (int t = 0; t < N; t++) {
            int bs = state[t].cols();
            if (PEEP) {
                for (int b = 0; b < bs; b++) {
                    if (t > 0) DWIP.A += gierr[t].col(b).A * state[t-1].col(b).A;
                    if (t > 0) DWFP.A += gferr[t].col(b).A * state[t-1].col(b).A;
                    DWOP.A += goerr[t].col(b).A * state[t].col(b).A;
                }
            }
            DWGI += gierr[t] * source[t].transpose();
            if (t > 0) DWGF += gferr[t] * source[t].transpose();
            DWGO += goerr[t] * source[t].transpose();
            DWCI += cierr[t] * source[t].transpose();
        }
        nsteps += N;
        nseq += 1;
    }
#undef A
    void gradient_clip(Sequence &s, Float m=1.0) {
        for (int t = 0; t < s.size(); t++) {
            s[t] = s[t].unaryExpr([m](Float x)
                                  { return x > m ? m : x < -m ? -m : x; });
        }
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else throw "unknown normalization";
        WGI += lr * DWGI;
        WGF += lr * DWGF;
        WGO += lr * DWGO;
        WCI += lr * DWCI;
        if (PEEP) {
            WIP += lr * DWIP;
            WFP += lr * DWFP;
            WOP += lr * DWOP;
        }
        DWGI *= momentum;
        DWGF *= momentum;
        DWGO *= momentum;
        DWCI *= momentum;
        if (PEEP) {
            DWIP *= momentum;
            DWFP *= momentum;
            DWOP *= momentum;
        }
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".WGI", &WGI, &DWGI);
        f(prefix+".WGF", &WGF, &DWGF);
        f(prefix+".WGO", &WGO, &DWGO);
        f(prefix+".WCI", &WCI, &DWCI);
        if (PEEP) {
            f(prefix+".WIP", &WIP, &DWIP);
            f(prefix+".WFP", &WFP, &DWFP);
            f(prefix+".WOP", &WOP, &DWOP);
        }
    }
    virtual void mystates(const string &prefix, StateFun f) {
        f(prefix+".inputs", &inputs);
        f(prefix+".d_inputs", &d_inputs);
        f(prefix+".outputs", &outputs);
        f(prefix+".d_outputs", &d_outputs);
        f(prefix+".state", &state);
        f(prefix+".stateerr", &stateerr);
        f(prefix+".gi", &gi);
        f(prefix+".gierr", &gierr);
        f(prefix+".go", &go);
        f(prefix+".goerr", &goerr);
        f(prefix+".gf", &gf);
        f(prefix+".gferr", &gferr);
        f(prefix+".ci", &ci);
        f(prefix+".cierr", &cierr);
    }
    Sequence *getState() {
        return &state;
    }
};

typedef GenericLSTM<SigmoidNonlin, TanhNonlin, TanhNonlin> LSTM;
REGISTER(LSTM);
typedef GenericLSTM<SigmoidNonlin, TanhNonlin, TanhNonlin, false> NPLSTM;
REGISTER(NPLSTM);
typedef GenericLSTM<SigmoidNonlin, TanhNonlin, NoNonlin> LINLSTM;
REGISTER(LINLSTM);
typedef GenericLSTM<SigmoidNonlin, ReluNonlin, TanhNonlin> RELUTANHLSTM;
REGISTER(RELUTANHLSTM);
typedef GenericLSTM<SigmoidNonlin, ReluNonlin, NoNonlin> RELULSTM;
REGISTER(RELULSTM);
typedef GenericLSTM<SigmoidNonlin, ReluNonlin, ReluNonlin> RELU2LSTM;
REGISTER(RELU2LSTM);

INetwork *make_LSTM() {
    return new LSTM();
}

struct LSTM1 : Stacked {
    LSTM1() {
        name = "lstm1";
        attributes["kind"] = "lstm1";
    }
    void initialize() {
        int ni = irequire("ninput");
        int nh = irequire("nhidden");
        int no = irequire("noutput");
        shared_ptr<INetwork> fwd, softmax;
        fwd = make_shared<LSTM>();
        fwd->init(nh, ni);
        add(fwd);
        softmax = make_shared<SoftmaxLayer>();
        softmax->init(no, nh);
        add(softmax);
    }
};
REGISTER(LSTM1);

INetwork *make_LSTM1() {
    return new LSTM1();
}

struct REVLSTM1 : Stacked {
    REVLSTM1() {
        name = "revlstm1";
    }
    void initialize() {
        int ni = irequire("ninput");
        int nh = irequire("nhidden");
        int no = irequire("noutput");
        shared_ptr<INetwork> fwd, rev, softmax;
        fwd = make_shared<LSTM>();
        fwd->init(nh, ni);
        rev = make_shared<Reversed>();
        rev->add(fwd);
        add(rev);
        softmax = make_shared<SoftmaxLayer>();
        softmax->init(no, nh);
        add(softmax);
    }
};
REGISTER(REVLSTM1);

INetwork *make_REVLSTM1() {
    return new REVLSTM1();
}

struct BIDILSTM : Stacked {
    BIDILSTM() {
        name = "bidilstm";
        attributes["kind"] = "bidi";
    }
    void initialize() {
        int ni = irequire("ninput");
        int nh = irequire("nhidden");
        int no = irequire("noutput");
        shared_ptr<INetwork> fwd, bwd, parallel, reversed, softmax;
        fwd = make_net(attr("lstm_type", "LSTM"));
        bwd = make_net(attr("lstm_type", "LSTM"));
        softmax = make_net(attr("output_type", "SoftmaxLayer"));
        fwd->init(nh, ni);
        bwd->init(nh, ni);
        reversed = make_shared<Reversed>();
        reversed->add(bwd);
        parallel = make_shared<Parallel>();
        parallel->add(fwd);
        parallel->add(reversed);
        add(parallel);
        softmax->init(no, 2*nh);
        add(softmax);
    }
};
REGISTER(BIDILSTM);

INetwork *make_BIDILSTM() {
    return new BIDILSTM();
}

INetwork *make_LRBIDILSTM() {
    INetwork *net = make_BIDILSTM();
    net->set("output_type", "SigmoidLayer");
    return net;
}
int status_LRBIDILSTM = ( network_factories["LRBIDILSTM"] = make_LRBIDILSTM, 0 );

INetwork *make_LINBIDILSTM() {
    INetwork *net = make_BIDILSTM();
    net->set("output_type", "LinearLayer");
    return net;
}
int status_LINBIDILSTM = ( network_factories["LINBIDILSTM"] = make_LINBIDILSTM, 0 );

struct BidiLayer : Parallel {
    BidiLayer() {
        name = "bidilayer";
    }
    void initialize() {
        int ni = irequire("ninput");
        int nh = irequire("noutput");
        INetwork *parallel = this;
        shared_ptr<INetwork> fwd, bwd, reversed;
        fwd = make_shared<LSTM>();
        fwd->init(nh, ni);
        bwd = make_shared<LSTM>();
        bwd->init(nh, ni);
        reversed = make_shared<Reversed>();
        reversed->add(bwd);
        parallel->add(fwd);
        parallel->add(reversed);
    }
};
REGISTER(BidiLayer);

shared_ptr<INetwork> make_fwdbwd(int nh, int ni) {
    shared_ptr<INetwork> net;
    net.reset(new BidiLayer());
    net->init(nh, ni);
    return net;
}

struct BIDILSTM2 : Stacked {
    BIDILSTM2() {
        name = "bidilstm2";
        attributes["kind"] = "bidi2";
    }
    void initialize() {
        int ni = irequire("ninput");
        int nh = irequire("nhidden");
        int nh2 = irequire("nhidden2");
        int no = irequire("noutput");
        // cerr << ">>> " << ni << " " << nh << " " << nh2 << " " << no << endl;
        shared_ptr<INetwork> parallel1, parallel2, logreg;
        parallel1 = make_fwdbwd(nh, ni);
        add(parallel1);
        parallel2 = make_fwdbwd(nh2, 2*nh);
        add(parallel2);
        logreg = make_shared<SoftmaxLayer>();
        logreg->init(no, 2*nh2);
        add(logreg);
    }
};
REGISTER(BIDILSTM2);

INetwork *make_BIDILSTM2() {
    return new BIDILSTM2();
}

inline Float log_add(Float x, Float y) {
    if (abs(x-y) > 10) return fmax(x, y);
    return log(exp(x-y)+1) + y;
}

inline Float log_mul(Float x, Float y) {
    return x+y;
}

void forward_algorithm(Mat &lr, Mat &lmatch, double skip) {
    int n = lmatch.rows(), m = lmatch.cols();
    lr.resize(n, m);
    Vec v(m), w(m);
    for (int j = 0; j < m; j++) v(j) = skip * j;
    for (int i = 0; i < n; i++) {
        w.segment(1, m-1) = v.segment(0, m-1);
        w(0) = skip * i;
        for (int j = 0; j < m; j++) {
            Float same = log_mul(v(j), lmatch(i, j));
            Float next = log_mul(w(j), lmatch(i, j));
            v(j) = log_add(same, next);
        }
        lr.row(i) = v;
    }
}

void forwardbackward(Mat &both, Mat &lmatch) {
    Mat lr;
    forward_algorithm(lr, lmatch);
    Mat rlmatch = lmatch;
    rlmatch = rlmatch.rowwise().reverse().eval();
    rlmatch = rlmatch.colwise().reverse().eval();
    Mat rl;
    forward_algorithm(rl, rlmatch);
    rl = rl.colwise().reverse().eval();
    rl = rl.rowwise().reverse().eval();
    both = lr + rl;
}

void ctc_align_targets(Mat &posteriors, Mat &outputs, Mat &targets) {
    double lo = 1e-5;
    int n1 = outputs.rows();
    int n2 = targets.rows();
    int nc = targets.cols();

    // compute log probability of state matches
    Mat lmatch;
    lmatch.resize(n1, n2);
    for (int t1 = 0; t1 < n1; t1++) {
        Vec out = outputs.row(t1);
        out = out.cwiseMax(lo);
        out /= out.sum();
        for (int t2 = 0; t2 < n2; t2++) {
            double value = out.transpose() * targets.row(t2).transpose();
            lmatch(t1, t2) = log(value);
        }
    }
    // compute unnormalized forward backward algorithm
    Mat both;
    forwardbackward(both, lmatch);

    // compute normalized state probabilities
    Mat epath = (both.array() - both.maxCoeff()).unaryExpr(ptr_fun(limexp));
    for (int j = 0; j < n2; j++) {
        double l = epath.col(j).sum();
        epath.col(j) /= l == 0 ? 1e-9 : l;
    }
    debugmat = epath;

    // compute posterior probabilities for each class and normalize
    Mat aligned;
    aligned.resize(n1, nc);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < nc; j++) {
            double total = 0.0;
            for (int k = 0; k < n2; k++) {
                double value = epath(i, k) * targets(k, j);
                total += value;
            }
            aligned(i, j) = total;
        }
    }
    for (int i = 0; i < n1; i++) {
        aligned.row(i) /= fmax(1e-9, aligned.row(i).sum());
    }

    posteriors = aligned;
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Sequence &targets) {
    assert(outputs[0].cols() == 1);
    assert(targets[0].cols() == 1);
    int n1 = outputs.size();
    int n2 = targets.size();
    int nc = targets[0].size();
    Mat moutputs(n1, nc);
    Mat mtargets(n2, nc);
    for (int i = 0; i < n1; i++) moutputs.row(i) = outputs[i].col(0);
    for (int i = 0; i < n2; i++) mtargets.row(i) = targets[i].col(0);
    Mat aligned;
    ctc_align_targets(aligned, moutputs, mtargets);
    posteriors.resize(n1);
    for (int i = 0; i < n1; i++) {
        posteriors[i].resize(aligned.row(i).size(), 1);
        posteriors[i].col(0) = aligned.row(i);
    }
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Classes &targets) {
    int nclasses = outputs[0].size();
    Sequence stargets;
    stargets.resize(targets.size());
    for (int t = 0; t < stargets.size(); t++) {
        stargets[t].resize(nclasses, 1);
        stargets[t].fill(0);
        stargets[t](targets[t], 0) = 1.0;
    }
    ctc_align_targets(posteriors, outputs, stargets);
}

void mktargets(Sequence &seq, Classes &transcript, int ndim) {
    seq.resize(2*transcript.size()+1);
    for (int t = 0; t < seq.size(); t++) {
        seq[t].setZero(ndim, 1);
        if (t%2 == 1) seq[t](transcript[(t-1)/2]) = 1;
        else seq[t](0) = 1;
    }
}

// Loading and saving networks; the logic of this is quite twisted and needs
// to be cleaned up. However, for now it works.

void save_codec(h5eigen::HDF5 *h5, const char *name, vector<int> &codec, int n) {
    assert(codec.size() == 0 || codec.size() == n);
    Vec vcodec;
    vcodec.resize(n);
    if (codec.size() == 0)
        for (int i = 0; i < n; i++) vcodec[i] = i;
    else
        for (int i = 0; i < n; i++) vcodec[i] = codec[i];

    h5->put(vcodec, name);
}

void load_codec(vector<int> &codec, h5eigen::HDF5 *h5, const char *name) {
    codec.resize(0);
    if (!h5->exists(name)) return;
    Vec vcodec;
    h5->get(vcodec, name);
    codec.resize(vcodec.size());
    for (int i = 0; i < vcodec.size(); i++) codec[i] = vcodec[i];
}

void save_net_raw(const char *fname, INetwork *net) {
    using namespace h5eigen;
    unique_ptr<HDF5> h5(make_HDF5());
    h5->open(fname, true);
    net->attributes["clstm-version"] = "1";
    for (auto &kv : net->attributes) {
    h5->setAttr(kv.first, kv.second);
    }
    save_codec(h5.get(), "codec", net->codec, net->noutput());
    save_codec(h5.get(), "icodec", net->icodec, net->ninput());

    net->weights("", [&h5](const string &prefix, VecMat a, VecMat da) {
                     if (a.mat) h5->put(*a.mat, prefix.c_str());
                     else if (a.vec) h5->put(*a.vec, prefix.c_str());
                     else throw "oops (save type)";
                 });
}

void load_net_raw(INetwork *net, const char *fname) {
    using namespace h5eigen;
    unique_ptr<HDF5> h5(make_HDF5());
    h5->open(fname);
    h5->getAttrs(net->attributes);
    load_codec(net->icodec, h5.get(), "icodec");
    load_codec(net->codec, h5.get(), "codec");
    net->weights("", [&h5](const string &prefix, VecMat a, VecMat da) {
                     if (a.mat) h5->get(*a.mat, prefix.c_str());
                     else if (a.vec) h5->get1d(*a.vec, prefix.c_str());
                     else throw "oops (load type)";
                 });
    net->networks("", [] (string s, INetwork *net) {
                      net->postLoad();
                  });
}

static string fixup_net_name(string kind) {
    if (kind == "bidi") return "BIDILSTM";
    if (kind == "lrbidi") return "LRBIDILSTM";
    if (kind == "bidi2") return "BIDILSTM2";
    if (kind == "lstm1") return "LSTM1";
    if (kind == "uni") return "LSTM1";
    return kind;
}

shared_ptr<INetwork> make_net(string kind) {
    kind = fixup_net_name(kind);
    auto it = network_factories.find(kind);
    if (it == network_factories.end()) throw "unknown network type";
    shared_ptr<INetwork> net;
    net.reset(it->second());
    net->attributes["kind"] = kind;
    return net;
}

void load_attributes(map<string, string> &attributes, const string &fname) {
    using namespace h5eigen;
    unique_ptr<HDF5> h5(make_HDF5());
    h5->open(fname.c_str());
    h5->getAttrs(attributes);
}

shared_ptr<INetwork> load_net(const string &fname) {
    bool verbose = (getenv("load_verbose") && atoi(getenv("load_verbose")));
    map<string, string> attributes;
    load_attributes(attributes, fname);
    string kind = attributes["kind"];
    if (verbose) {
        cout << "loading " << fname << endl;
        cout << endl;
        cout << "attributes:" << endl;
        for (auto kv : attributes)
            cout << "   " << kv.first << " = " << kv.second << endl;
        cout << endl;
    }
    shared_ptr<INetwork> net;
    net = make_net(kind);
    net->attributes = attributes;
    net->initialize();
    load_net_raw(net.get(), fname.c_str());
    if (verbose) {
        cout << "network:" << endl;
        net->info("");
        cout << endl;
    }
    return net;
}

void save_net(const string &fname, shared_ptr<INetwork> net) {
    rename(fname.c_str(), (fname+"~").c_str());
    save_net_raw(fname.c_str(), net.get());
}

void trivial_decode(Classes &cs, Sequence &outputs, int batch) {
    int N = outputs.size();
    int t = 0;
    float mv = 0;
    int mc = -1;
    while (t < N) {
        int index;
        float v = outputs[t].col(batch).maxCoeff(&index);
        if (index == 0) {
            // NB: there should be a 0 at the end anyway
            if (mc != -1 && mc != 0) cs.push_back(mc);
            mv = 0; mc = -1; t++;
            continue;
        }
        if (v > mv) {
            mv = v;
            mc = index;
        }
        t++;
    }
}

void ctc_train(INetwork *net, Sequence &xs, Sequence &targets) {
    // untested
    assert(xs[0].cols() == 1);
    assert(xs.size() <= targets.size());
    assert(!anynan(xs));
    net->inputs = xs;
    net->forward();
    if (anynan(net->outputs)) throw "got NaN";
    Sequence aligned;
    ctc_align_targets(aligned, net->outputs, targets);
    if (anynan(aligned)) throw "got NaN";
    set_targets(net, aligned);
    net->backward();
}

void ctc_train(INetwork *net, Sequence &xs, Classes &targets) {
    // untested
    Sequence ys;
    mktargets(ys, targets, net->noutput());
    ctc_train(net, xs, ys);
}

void ctc_train(INetwork *net, Sequence &xs, BatchClasses &targets) {
    throw "unimplemented";
}
}  // namespace ocropus

#ifdef LSTM_TEST
// We include the test cases in the source file because we want
// direct access to internal variables from the test cases.
#include "lstm_test.i"
#endif
