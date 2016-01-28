// -*- C++ -*-

// Additional functions and utilities for CLSTM networks.
// These may use the array classes from "multidim.h"

#ifndef ocropus_clstm_utils_
#define ocropus_clstm_utils_

#include <glob.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "pstring.h"

namespace ocropus {
using std::string;
using std::wstring;
using std::vector;
using std::istream;
using std::ostream;
using std::ifstream;
using std::ofstream;
using std::endl;
using std::cout;
using std::cerr;

template <class A>
inline void die(const A &arg) {
  cerr << "EXCEPTION (" << arg << ")\n";
  exit(255);
}

// get current time down to usec precision as a double

inline double now() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

inline void glob(vector<string> &result, const string &arg) {
  result.clear();
  glob_t buf;
  glob(arg.c_str(), GLOB_TILDE, nullptr, &buf);
  for (int i = 0; i < buf.gl_pathc; i++) {
    result.push_back(buf.gl_pathv[i]);
  }
  if (buf.gl_pathc > 0) globfree(&buf);
}

inline string basename(string s) {
  int start = 0;
  for (;;) {
    auto pos = s.find("/", start);
    if (pos == string::npos) break;
    start = pos + 1;
  }
  auto pos = s.find(".", start);
  if (pos == string::npos)
    return s;
  else
    return s.substr(0, pos);
}

inline string read_text(string fname, int maxsize = 65536) {
  vector<char> buf_v(maxsize);
  char *buf = &buf_v[0];
  buf[maxsize - 1] = 0;
  ifstream stream(fname);
  stream.read(buf, maxsize - 1);
  int n = stream.gcount();
  while (n > 0 && buf[n - 1] == '\n') n--;
  return string(buf, n);
}

inline wstring read_text32(string fname, int maxsize = 65536) {
  vector<char> buf_v(maxsize);
  char *buf = &buf_v[0];
  buf[maxsize - 1] = 0;
  ifstream stream(fname);
  stream.read(buf, maxsize - 1);
  int n = stream.gcount();
  while (n > 0 && buf[n - 1] == '\n') n--;
  return utf8_to_utf32(string(buf, n));
}

inline void read_lines(vector<string> &lines, string fname) {
  ifstream stream(fname);
  string line;
  lines.clear();
  while (getline(stream, line)) {
    lines.push_back(line);
  }
}

inline void write_text(const string fname, const wstring &data) {
  string utf8 = utf32_to_utf8(data);
  ofstream stream(fname);
  stream << utf8 << endl;
}

inline void write_text(const string fname, const string &data) {
  ofstream stream(fname);
  stream << data << endl;
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

#define PRINT(...) print(__FILE__, __LINE__, __VA_ARGS__)

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

bool reported_params(const char *name);

template <class T>
inline void report_params(const char *name, const T &value) {
  const char *flag = getenv("params");
  if (flag && !atoi(flag)) return;
  if (reported_params(name)) return;
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

// A class encapsulating "report every ..." type logic.
// This will generally report every `every` steps, as well
// as when the `upto` value is reached. It can be disabled
// by setting `enabled` to false.

struct Trigger {
  bool finished = false;
  bool enabled = true;
  int count = 0;
  int every = 1;
  int upto = 0;
  int next = 0;
  int last_trigger = 0;
  int current_trigger = 0;
  Trigger(int every, int upto = -1, int start = 0)
      : count(start), every(every), upto(upto) {}
  Trigger &skip0() {
    next += every;
    return *this;
  }
  Trigger &enable(bool flag) {
    enabled = flag;
    return *this;
  }
  void rotate() {
    last_trigger = current_trigger;
    current_trigger = count;
  }
  int since() { return count - last_trigger; }
  bool check() {
    assert(!finished);
    if (upto > 0 && count >= upto - 1) {
      finished = true;
      rotate();
      return true;
    }
    if (every == 0) return false;
    if (count >= next) {
      while (count >= next) next += every;
      rotate();
      return true;
    } else {
      return false;
    }
  }
  bool operator()(int current) {
    assert(!finished);
    assert(current >= count);
    count = current;
    return check();
  }
  bool operator+=(int incr) { return operator()(count + incr); }
  bool operator++() { return operator()(count + 1); }
};
}

#endif
