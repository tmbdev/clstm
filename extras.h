// -*- C++ -*-

// Additional functions and utilities for CLSTM networks.
// These may use the array classes from "multidim.h"

#ifndef ocropus_clstm_extras_
#define ocropus_clstm_extras_

#include "clstm.h"
#include "multidim.h"
#include <string>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <map>
#include <stdarg.h>
#include <glob.h>
#include "pstring.h"

#include <iostream>

template <class T>
inline std::ostream &operator<<(std::ostream &stream, multidim::mdarray<T> &a) {
  if (a.rank() == 1) {
    stream << "[mdarray1d";
    for (int i = 0; i < a.dim(0); i++) stream << " " << a(i);
    stream << "]" << std::endl;
  } else if (a.rank() == 2) {
    stream << "[mdarray2d" << std::endl;
    for (int i = 0; i < a.dim(0); i++) {
      for (int j = 0; j < a.dim(1); j++) {
        stream << " " << a(i);
      }
      stream << std::endl;
    }
    stream << "]" << std::endl;
  } else {
    stream << "[mdarray of rank > 2]" << std::endl;
  }
  return stream;
}

namespace ocropus {
using std::string;
using std::wstring;
using std::shared_ptr;
using std::vector;
using std::cout;
using std::ostream;
using std::cerr;
using std::endl;
using std::min;
using namespace multidim;

void glob(vector<string> &result, const string &arg);

void srandomize();
unsigned urandom();
int irandom();
double drandom();

// simplistic sprintf for strings

inline string stringf(const char *format, ...) {
  static char buf[4096];
  va_list v;
  va_start(v, format);
  vsnprintf(buf, sizeof(buf), format, v);
  va_end(v);
  return string(buf);
}

// get current time down to usec precision as a double

inline double now() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// print the arguments to cout

inline void print() { cout << endl; }

inline ostream &operator<<(ostream &stream, const std::wstring &arg) {
  cout << utf32_to_utf8(arg);
  return stream;
}

template <class T>
inline void print(const T &arg) {
  using namespace std;
  cout << arg << endl;
}

template <class T, typename... Args>
inline void print(T arg, Args... args) {
  cout << arg << " ";
  print(args...);
}

inline string getdef(std::map<string, string> &m, const string &key,
                     const string &dflt) {
  auto it = m.find(key);
  if (it == m.end()) return dflt;
  return it->second;
}

// print the arguments to cerr

inline void dprint() { cerr << endl; }

template <class T>
inline void dprint(const T &arg) {
  cerr << arg << endl;
}

template <class T, typename... Args>
inline void dprint(T arg, Args... args) {
  cerr << arg << " ";
  dprint(args...);
}

// get values from the environment, with defaults

template <class T>
inline void report_params(const char *name, const T &value) {
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

inline int split(vector<string> &tokens, string s, char c = ':') {
  int last = 0;
  for (;;) {
    size_t next = s.find(c, last);
    if (next == string::npos) {
      tokens.push_back(s.substr(last));
      break;
    }
    tokens.push_back(s.substr(last, next - last));
    last = next + 1;
  }
  return tokens.size();
}

inline string getoneof(const char *name, const char *dflt) {
  string s = dflt;
  if (getenv(name)) s = getenv(name);
  vector<string> tokens;
  int n = split(tokens, s);
  int k = (irandom() / 1792) % n;
  // cerr << "# getoneof " << name << " " << n << " " << k << endl;
  string result = tokens[k];
  report_params(name, result);
  return result;
}

inline int getienv(const char *name, int dflt = 0) {
  int result = dflt;
  if (getenv(name)) result = atoi(getenv(name));
  report_params(name, result);
  return result;
}

inline double getdenv(const char *name, double dflt = 0) {
  double result = dflt;
  if (getenv(name)) result = atof(getenv(name));
  report_params(name, result);
  return result;
}

// get a value or random value from the environment (var=7.3 or var=2,8)

inline double getrenv(const char *name, double dflt = 0, bool logscale = true) {
  const char *s = getenv(name);
  if (!s) return dflt;
  float lo, hi;
  if (sscanf(s, "%g,%g", &lo, &hi) == 2) {
    double x = exp(log(lo) + drandom() * (log(hi) - log(lo)));
    report_params(name, x);
    return x;
  } else if (sscanf(s, "%g", &lo) == 1) {
    report_params(name, lo);
    return lo;
  } else {
    throwf("bad format for getrenv");
    return 0;
  }
}

inline double getuenv(const char *name, double dflt = 0) {
  const char *s = getenv(name);
  if (!s) return dflt;
  float lo, hi;
  if (sscanf(s, "%g,%g", &lo, &hi) == 2) {
    double x = lo + drandom() * (hi - lo);
    report_params(name, x);
    return x;
  } else if (sscanf(s, "%g", &lo) == 1) {
    report_params(name, lo);
    return lo;
  } else {
    throwf("bad format for getuenv");
    return 0;
  }
}

// array minimum, maximum

template <class T>
T amin(mdarray<T> &a) {
  T m = a[0];
  for (int i = 1; i < a.size(); i++)
    if (a[i] < m) m = a[i];
  return m;
}

template <class T>
T amax(mdarray<T> &a) {
  T m = a[0];
  for (int i = 1; i < a.size(); i++)
    if (a[i] > m) m = a[i];
  return m;
}

// text line normalization

struct INormalizer {
  int target_height = 48;
  float smooth2d = 1.0;
  float smooth1d = 0.3;
  float range = 4.0;
  float vscale = 1.0;
  virtual ~INormalizer() {}
  virtual void getparams(bool verbose = false) {}
  virtual void measure(mdarray<float> &line) = 0;
  virtual void normalize(mdarray<float> &out, mdarray<float> &in) = 0;
  virtual void setPyServer(void *p) {}
};

INormalizer *make_Normalizer(const string &);
INormalizer *make_NoNormalizer();
INormalizer *make_MeanNormalizer();
INormalizer *make_CenterNormalizer();

void read_png(mdarray<unsigned char> &image, FILE *fp, bool gray = false);
void write_png(FILE *fp, mdarray<unsigned char> &image);
void read_png(mdarray<unsigned char> &image, const char *name,
              bool gray = false);
void write_png(const char *name, mdarray<unsigned char> &image);

void read_png(mdarray<float> &image, FILE *fp, bool gray = false);
void write_png(FILE *fp, mdarray<float> &image);
void read_png(mdarray<float> &image, const char *name, bool gray = false);
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
    for (int j = 0; j < dest.dim(1); j++) dest(i, j) = src(j, i);
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
Network make_net_init(const string &kind, int nclasses, int dim,
                      string prefix = "");

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
