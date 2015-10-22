// -*- C++ -*-

// Additional functions and utilities for CLSTM networks.
// These may use the array classes from "multidim.h"

#ifndef ocropus_clstm_utils_
#define ocropus_clstm_utils_

#include "clstm.h"
#include <string>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <map>
#include <stdarg.h>
#include <glob.h>
#include "pstring.h"
#include <iostream>

namespace ocropus {

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
  int k = (lrand48() / 1792) % n;
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
    double x = exp(log(lo) + drand48() * (log(hi) - log(lo)));
    report_params(name, x);
    return x;
  } else if (sscanf(s, "%g", &lo) == 1) {
    report_params(name, lo);
    return lo;
  } else {
    THROW("bad format for getrenv");
    return 0;
  }
}

inline double getuenv(const char *name, double dflt = 0) {
  const char *s = getenv(name);
  if (!s) return dflt;
  float lo, hi;
  if (sscanf(s, "%g,%g", &lo, &hi) == 2) {
    double x = lo + drand48() * (hi - lo);
    report_params(name, x);
    return x;
  } else if (sscanf(s, "%g", &lo) == 1) {
    report_params(name, lo);
    return lo;
  } else {
    THROW("bad format for getuenv");
    return 0;
  }
}

inline string stringf(const char *format, ...) {
  static char buf[4096];
  va_list v;
  va_start(v, format);
  vsnprintf(buf, sizeof(buf), format, v);
  va_end(v);
  return string(buf);
}

inline void throwf(const char *format, ...) {
  static char buf[1024];
  va_list arglist;
  va_start(arglist, format);
  vsprintf(buf, format, arglist);
  va_end(arglist);
  THROW(buf);
}

}

#endif
