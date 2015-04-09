// -*- C++ -*-

// Additional functions and utilities for CLSTM networks.
// These may use the array classes from "multidim.h"

#ifndef clstm_extras_
#define clstm_extras_

#include "clstm.h"
#include "multidim.h"
#include <string>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <map>

namespace ocropus {
using std::string;
using std::wstring;
using std::shared_ptr;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using namespace multidim;

void srandomize();
unsigned urandom();
int irandom();
double drandom();

// get current time down to usec precision as a double

inline double now() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// print the arguments to cout

template <class T>
inline void print(const T &arg) {
    cout << arg << endl;
}

template <class T, typename ... Args>
inline void print(T arg, Args ... args) {
    cout << arg << " ";
    print(args ...);
}

inline string getdef(std::map<string, string> &m, const string &key, const string &dflt) {
    auto it = m.find(key);
    if (it == m.end()) return dflt;
    return it->second;
}

// print the arguments to cerr

template <class T>
inline void dprint(const T &arg) {
    cerr << arg << endl;
}

template <class T, typename ... Args>
inline void dprint(T arg, Args ... args) {
    cerr << arg << " ";
    dprint(args ...);
}

// get values from the environment, with defaults

template <class T>
inline void report_params(const char *name, const T &value ) {
    const char *flag = getenv("params");
    if (!flag || !atoi(flag)) return;
    cerr << "#: " << name << " = " << value << endl;
}

inline const char *getsenv(const char *name, const char *dflt) {
    const char *result = dflt;
    if (getenv(name)) result = getenv(name);
    report_params(name, result);
    return result;
}

inline int split(vector<string> &tokens, string s, char c=':') {
    int last = 0;
    for (;; ) {
        size_t next = s.find(c, last);
        if (next == string::npos) {
            tokens.push_back(s.substr(last));
            break;
        }
        tokens.push_back(s.substr(last, next-last));
        last = next+1;
    }
    return tokens.size();
}

inline string getoneof(const char *name, const char *dflt) {
    string s = dflt;
    if (getenv(name)) s = getenv(name);
    vector<string> tokens;
    int n = split(tokens, s);
    int k = (irandom()/1792)%n;
    // cerr << "# getoneof " << name << " " << n << " " << k << endl;
    string result = tokens[k];
    report_params(name, result);
    return result;
}

inline int getienv(const char *name, int dflt=0) {
    int result = dflt;
    if (getenv(name)) result = atoi(getenv(name));
    report_params(name, result);
    return result;
}

inline double getdenv(const char *name, double dflt=0) {
    double result = dflt;
    if (getenv(name)) result = atof(getenv(name));
    report_params(name, result);
    return result;
}

// get a value or random value from the environment (var=7.3 or var=2,8)

inline double getrenv(const char *name, double dflt=0, bool logscale=true) {
    const char *s = getenv(name);
    if (!s) return dflt;
    float lo, hi;
    if (sscanf(s, "%g,%g", &lo, &hi) == 2) {
        double x = exp(log(lo)+drandom()*(log(hi)-log(lo)));
        report_params(name, x);
        return x;
    } else if (sscanf(s, "%g", &lo) == 1) {
        report_params(name, lo);
        return lo;
    } else {
        throw "bad format for getrenv";
    }
}

inline double getuenv(const char *name, double dflt=0) {
    const char *s = getenv(name);
    if (!s) return dflt;
    float lo, hi;
    if (sscanf(s, "%g,%g", &lo, &hi) == 2) {
        double x = lo+drandom()*(hi-lo);
        report_params(name, x);
        return x;
    } else if (sscanf(s, "%g", &lo) == 1) {
        report_params(name, lo);
        return lo;
    } else {
        throw "bad format for getuenv";
    }
}

// array minimum, maximum

template <class T>
T amin(mdarray<T> &a) {
    T m = a[0];
    for (int i = 1; i < a.size(); i++) if (a[i] < m) m = a[i];
    return m;
}

template <class T>
T amax(mdarray<T> &a) {
    T m = a[0];
    for (int i = 1; i < a.size(); i++) if (a[i] > m) m = a[i];
    return m;
}

// text line normalization

struct INormalizer {
    int target_height = 48;
    float smooth2d = 1.0;
    float smooth1d = 0.3;
    float range = 4.0;
    float vscale = 1.0;
    virtual ~INormalizer() {
    }
    virtual void getparams(bool verbose=false) {
    }
    virtual void measure(mdarray<float> &line) = 0;
    virtual void normalize(mdarray<float> &out, mdarray<float> &in) = 0;
    virtual void setPyServer(void *p) {
    }
};

INormalizer *make_Normalizer(const string &);
INormalizer *make_NoNormalizer();
INormalizer *make_MeanNormalizer();
INormalizer *make_CenterNormalizer();

// OCR dataset access, including datasets that are normalized
// on the fly

struct IOcrDataset {
    virtual ~IOcrDataset() {
    }
    virtual void image(mdarray<float> &a, int index) = 0;
    virtual void transcript(mdarray<int> &a, int index) = 0;
    virtual string to_string(mdarray<int> &transcript) = 0;
    virtual string to_string(vector<int> &transcript) = 0;
    virtual void getCodec(vector<int> &codec) = 0;
    virtual int samples() = 0;
    virtual int dim() = 0;
    virtual int classes() = 0;
};

IOcrDataset *make_HDF5Dataset(const string &fname, bool varsize=false);
IOcrDataset *make_NormalizedDataset(shared_ptr<IOcrDataset> &dataset,
                                    shared_ptr<INormalizer> &normalizer);
IOcrDataset *make_Dataset(const string &fname);

void read_png(mdarray<unsigned char> &image, FILE *fp, bool gray=false);
void write_png(FILE *fp, mdarray<unsigned char> &image);
void read_png(mdarray<unsigned char> &image, const char *name, bool gray=false);
void write_png(const char *name, mdarray<unsigned char> &image);

void read_png(mdarray<float> &image, FILE *fp, bool gray=false);
void write_png(FILE *fp, mdarray<float> &image);
void read_png(mdarray<float> &image, const char *name, bool gray=false);
void write_png(const char *name, mdarray<float> &image);

inline bool anynan(mdarray<float> &a) {
    for (int i = 0; i < a.size(); i++)
        if (isnan(a[i])) return true;
    return false;
}

template <class S, class T>
inline void assign(S &dest, T &src) {
    dest.resize_(src.dims);
    int n = dest.size();
    for (int i = 0; i < n; i++) dest.data[i] = src.data[i];
}

inline void assign(mdarray<int> &dest, vector<int> &src) {
    int n = src.size();
    dest.resize(n);
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

inline void assign(vector<int> &dest, mdarray<int> &src) {
    int n = src.dim(0);
    dest.resize(n);
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

template <class S, class T>
inline void transpose(S &dest, T &src) {
    dest.resize(src.dim(1), src.dim(0));
    for (int i = 0; i < dest.dim(0); i++)
        for (int j = 0; j < dest.dim(1); j++)
            dest(i, j) = src(j, i);
}

template <class T>
inline void transpose(T &a) {
    T temp;
    transpose(temp, a);
    assign(a, temp);
}

template <class T>
inline void assign(Sequence &seq, T &a) {
    assert(a.rank() == 2);
    seq.resize(a.dim(0));
    for (int t = 0; t < a.dim(0); t++) {
        seq[t].resize(a.dim(1), 1);
        for (int i = 0; i < a.dim(1); i++) seq[t](i, 0) = a(t, i);
    }
}

template <class T>
inline void assign(T &a, Sequence &seq) {
    a.resize(int(seq.size()), int(seq[0].size()));
    for (int t = 0; t < a.dim(0); t++) {
        for (int i = 0; i < a.dim(1); i++) a(t, i) = seq[t](i);
    }
}

template <class A, class T>
inline int indexof(A &a, const T &t) {
    for (int i = 0; i < a.size(); i++)
        if (a[i] == t) return i;
    return -1;
}

// simple network creation; this takes parameters from the environment
shared_ptr<INetwork> make_net_init(const string &kind, int nclasses, int dim, string prefix="");

// setting inputs and outputs
void set_inputs(INetwork *net, mdarray<float> &inputs);
void set_targets(INetwork *net, mdarray<float> &targets);
void set_targets_accelerated(INetwork *net, mdarray<float> &targets);
void set_classes(INetwork *net, mdarray<int> &targets);

// single sequence training functions
void mktargets(mdarray<float> &seq, mdarray<int> &targets, int ndim);
void train(INetwork *net, mdarray<float> &inputs, mdarray<float> &targets);
void ctrain(INetwork *net, mdarray<float> &inputs, mdarray<int> &targets);
void cpred(INetwork *net, mdarray<int> &preds, mdarray<float> &inputs);
void ctc_train(INetwork *net, mdarray<float> &xs, mdarray<float> &targets);
void ctc_train(INetwork *net, mdarray<float> &xs, mdarray<int> &targets);
void ctc_train(INetwork *net, mdarray<float> &xs, string &targets);
void ctc_train(INetwork *net, mdarray<float> &xs, wstring &targets);
}

#endif
