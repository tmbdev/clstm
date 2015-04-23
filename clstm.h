// -*- C++ -*-

// A basic LSTM implementation in C++. All you should need is clstm.cc and
// clstm.h. Library dependencies are limited to a small subset of STL and
// Eigen/Dense

#ifndef ocropus_lstm__
#define ocropus_lstm__

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <memory>
#include <map>
#include <Eigen/Dense>

namespace ocropus {
using std::string;
using std::vector;
using std::map;
using std::shared_ptr;
using std::unique_ptr;
using std::function;
using Eigen::Ref;

#ifdef LSTM_DOUBLE
typedef double Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
#else
typedef float Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXf Vec;
typedef Eigen::MatrixXf Mat;
#endif

typedef vector<Mat> Sequence;
typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

extern char exception_message[256];

inline Vec timeslice(const Sequence &s, int i, int b=0) {
    Vec result(s.size());
    for (int t = 0; t < s.size(); t++)
        result[t] = s[t](i, b);
    return result;
}

struct VecMat {
    Vec *vec = 0;
    Mat *mat = 0;
    VecMat() {
    }
    VecMat(Vec *vec) {
        this->vec = vec;
    }
    VecMat(Mat *mat) {
        this->mat = mat;
    }
};

struct ITrainable {
    virtual ~ITrainable() {
    }
    // Each network has a name that's used for loading
    // and saving.
    string name = "???";

    // Learning rate and momentum used for training.
    Float learning_rate = 1e-4;
    Float momentum = 0.9;
    enum Normalization : int {
        NORM_NONE, NORM_LEN, NORM_BATCH, NORM_DFLT = NORM_NONE,
    } normalization = NORM_DFLT;

    // The attributes array contains parameters for constructing the
    // network, as well as information necessary for loading and saving
    // networks.
    map<string, string> attributes;
    string attr(string key, string dflt="") {
        auto it = attributes.find(key);
        if (it == attributes.end()) return dflt;
        return it->second;
    }
    int iattr(string key, int dflt=-1) {
        auto it = attributes.find(key);
        if (it == attributes.end()) return dflt;
        return std::stoi(it->second);
    }
    int irequire(string key) {
        auto it = attributes.find(key);
        if (it == attributes.end()) {
            sprintf(exception_message, "missing parameter: %s", key.c_str());
            throw exception_message;
        }
        return std::stoi(it->second);
    }
    void set(string key, string value) {
        attributes[key] = value;
    }
    void set(string key, int value) {
        attributes[key] = std::to_string(value);
    }
    void set(string key, double value) {
        attributes[key] = std::to_string(value);
    }

    // Learning rates
    virtual void setLearningRate(Float lr, Float momentum) = 0;

    // Main methods for forward and backward propagation
    // of activations.
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void update() = 0;

    virtual int idepth() {
        return -9999;
    }
    virtual int odepth() {
        return -9999;
    }

    virtual void initialize() {
        // this gets initialization parameters
        // out of the attributes array
    }

    // These are convenience functions for initialization
    virtual void init(int no, int ni) final {
        set("ninput", ni);
        set("noutput", no);
        initialize();
    }
    virtual void init(int no, int nh, int ni) final {
        set("ninput", ni);
        set("nhidden", nh);
        set("noutput", no);
        initialize();
    }
    virtual void init(int no, int nh2, int nh, int ni) final {
        set("ninput", ni);
        set("nhidden", nh);
        set("nhidden2", nh2);
        set("noutput", no);
        initialize();
    }
};

struct INetwork : virtual ITrainable {
    // Networks have input and output "ports" for sequences
    // and derivatives. These are propagated in forward()
    // and backward() methods.
    Sequence inputs, d_inputs;
    Sequence outputs, d_outputs;

    // Some networks have subnetworks. They should be
    // stored in the `sub` vector. That way, functions
    // like `save` can automatically traverse the tree
    // of networks. Together with the `name` field,
    // this forms a hierarchical namespace of networks.
    vector<shared_ptr<INetwork> > sub;

    // Data for encoding/decoding input/output strings.
    vector<int> codec;
    vector<int> icodec;
    unique_ptr<map<int, int> > encoder;  // cached
    unique_ptr<map<int, int> > iencoder;  // cached
    void makeEncoders();
    std::wstring decode(Classes &cs);
    std::wstring idecode(Classes &cs);
    void encode(Classes &cs, std::wstring &s);
    void iencode(Classes &cs, std::wstring &s);

    // Parameters specific to softmax.
    Float softmax_floor = 1e-5;
    bool softmax_accel = false;

    virtual ~INetwork() {
    }

    // Expected number of input/output features.
    virtual int ninput() {
        return -999999;
    }
    virtual int noutput() {
        return -999999;
    }

    // Add a network as a subnetwork.
    virtual void add(shared_ptr<INetwork> net) {
        sub.push_back(net);
    }

    // Hooks to iterate over the weights and states of this network.
    typedef function<void (const string &, VecMat, VecMat)> WeightFun;
    typedef function<void (const string &, Sequence *)> StateFun;
    virtual void myweights(const string &prefix, WeightFun f) {
    }
    virtual void mystates(const string &prefix, StateFun f) {
    }

    // Hooks executed prior to saving and after loading.
    // Loading iterates over the weights with the `weights`
    // methods and restores only the weights. `postLoad`
    // allows classes to update other internal state that
    // depends on matrix size.
    virtual void preSave() {
    }
    virtual void postLoad() {
    }

    // Set the learning rate for this network and all subnetworks.
    virtual void setLearningRate(Float lr, Float momentum) {
        this->learning_rate = lr;
        this->momentum = momentum;
        for (int i = 0; i < sub.size(); i++)
            sub[i]->setLearningRate(lr, momentum);
    }

    void info(string prefix);
    void weights(const string &prefix, WeightFun f);
    void states(const string &prefix, StateFun f);
    void networks(const string &prefix, function<void (string, INetwork*)>);
    Sequence *getState(string name);
    // special method for LSTM and similar networks, returning the
    // primary internal state sequence
    Sequence *getState() {
        throw "unimplemented";
    };
    void save(const char *fname);
    void load(const char *fname);
};

// setting inputs and outputs
void set_inputs(INetwork *net, Sequence &inputs);
void set_targets(INetwork *net, Sequence &targets);
void set_targets_accelerated(INetwork *net, Sequence &targets);
void set_classes(INetwork *net, Classes &classes);
void set_classes(INetwork *net, BatchClasses &classes);

// single sequence training functions
void train(INetwork *net, Sequence &xs, Sequence &targets);
void ctrain(INetwork *net, Sequence &xs, Classes &cs);
void ctrain_accelerated(INetwork *net, Sequence &xs, Classes &cs, Float lo=1e-5);
void cpred(INetwork *net, Classes &preds, Sequence &xs);
void mktargets(Sequence &seq, Classes &targets, int ndim);

// batch training functions
void ctrain(INetwork *net, Sequence &xs, BatchClasses &cs);
void ctrain_accelerated(INetwork *net, Sequence &xs, BatchClasses &cs, Float lo=1e-5);
void cpred(INetwork *net, BatchClasses &preds, Sequence &xs);
void mktargets(Sequence &seq, BatchClasses &targets, int ndim);

// common network layers
INetwork *make_LinearLayer();
INetwork *make_LogregLayer();
INetwork *make_SoftmaxLayer();
INetwork *make_TanhLayer();
INetwork *make_ReluLayer();
INetwork *make_Stacked();
INetwork *make_Reversed();
INetwork *make_Parallel();

// prefab networks
INetwork *make_MLP();
INetwork *make_LSTM();
INetwork *make_LSTM1();
INetwork *make_REVLSTM1();
INetwork *make_BIDILSTM();
INetwork *make_BIDILSTM2();
INetwork *make_LRBIDILSTM();

typedef std::function<INetwork*(void)> INetworkFactory;
extern map<string, INetworkFactory> network_factories;
shared_ptr<INetwork> make_net(string kind);

extern Mat debugmat;

// loading and saving networks
void load_attributes(map<string, string> &attrs, const string &file);
shared_ptr<INetwork> load_net(const string &file);
void save_net(const string &file, shared_ptr<INetwork> net);

// training with CTC
void forward_algorithm(Mat &lr, Mat &lmatch, double skip=-5.0);
void forwardbackward(Mat &both, Mat &lmatch);
void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Sequence &targets);
void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Classes &targets);
void trivial_decode(Classes &cs, Sequence &outputs, int batch=0);
void ctc_train(INetwork *net, Sequence &xs, Sequence &targets);
void ctc_train(INetwork *net, Sequence &xs, Classes &targets);
void ctc_train(INetwork *net, Sequence &xs, BatchClasses &targets);
}

namespace {
inline bool anynan(ocropus::Sequence &a) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].rows(); j++) {
            for (int k = 0; k < a[i].cols(); k++) {
                if (isnan(a[i](j, k))) return true;
            }
        }
    }
    return false;
}

template <class A, class B>
double levenshtein(A &a, B &b) {
    using std::vector;
    int n = a.size();
    int m = b.size();
    if (n > m) return levenshtein(b, a);
    vector<double> current(n+1);
    vector<double> previous(n+1);
    for (int k = 0; k < current.size(); k++) current[k] = k;
    for (int i = 1; i <= m; i++) {
        previous = current;
        for (int k = 0; k < current.size(); k++) current[k] = 0;
        current[0] = i;
        for (int j = 1; j <= n; j++) {
            double add = previous[j]+1;
            double del = current[j-1]+1;
            double change = previous[j-1];
            if (a[j-1] != b[i-1]) change = change+1;
            current[j] = fmin(fmin(add, del), change);
        }
    }
    return current[n];
}
}

#endif
