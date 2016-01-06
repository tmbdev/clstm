// -*- C++ -*-

#ifndef pymulti_
#define pymulti_

#ifndef NODISPLAY
#include <zmqpp/zmqpp.hpp>
#endif
#include <stdarg.h>
#include <iostream>
#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

namespace pytensor {
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::cout;
using std::cerr;
using std::endl;

template <class T, size_t n>
using Tensor = Eigen::Tensor<T, n>;
template <class T, size_t n>
using TensorRM = Eigen::Tensor<T, n, Eigen::RowMajor>;

template <class T, size_t n>
void assign(Tensor<T, n> &dest, TensorRM<T, n> &src) {
  Eigen::array<int, n> rev;
  for (int i = 0; i < n; i++) rev[i] = n - i - 1;
  dest = src.swap_layout().shuffle(rev);
}
template <class T, size_t n>
void assign(TensorRM<T, n> &dest, Tensor<T, n> &src) {
  Eigen::array<int, n> rev;
  for (int i = 0; i < n; i++) rev[i] = n - i - 1;
  dest = src.swap_layout().shuffle(rev);
}

inline string stringf(const char *format, ...) {
  static char buf[4096];
  va_list v;
  va_start(v, format);
  vsnprintf(buf, sizeof(buf), format, v);
  va_end(v);
  return string(buf);
}

#ifdef NODISPLAY
struct PyServer {
  void open(const char *where = "tcp://127.0.0.1:9876") {}
  void setMode(int mode) {}
  string eval(string s) { return ""; }
  string eval(string s, const float *a, int na) { return ""; }
  string eval(string s, const float *a, int na, const float *b, int nb) {
    return "";
  }
  string evalf(const char *format, ...) { return ""; }
  void clf() {}
  void subplot(int rows, int cols, int n) {}
  void plot(Tensor<float, 1> &v, string extra = "") {}
  void plot2(Tensor<float, 1> &u, Tensor<float, 1> &v, string extra = "") {}
  void imshow(Tensor<float, 2> &a, string extra = "") {}
  void imshowT(Tensor<float, 2> &a, string extra = "") {}
};
#else
struct PyServer {
  int mode = 0;  // -1=ignore, 0=uninit, 1=output
  zmqpp::context context;
  unique_ptr<zmqpp::socket> socket;
  void open(const char *where = "tcp://127.0.0.1:9876") {
    if (string(where) == "none") {
      mode = -1;
      return;
    }
    socket.reset(new zmqpp::socket(context, zmqpp::socket_type::req));
    string addr = getenv("PYSERVER") ? getenv("PYSERVER") : where;
    cerr << "waiting for python server at " << addr << endl;
    socket->connect(addr.c_str());
    mode = 1;
    eval("print 'OK'");
    cerr << "connected" << endl;
    eval("from pylab import *");
    eval("ion()");
  }
  void setMode(int mode) { this->mode = mode; }
  string eval(string s) {
    if (mode < 0)
      return "";
    else if (mode < 1)
      THROW("uninitialized");
    zmqpp::message message;
    message << s;
    socket->send(message);
    socket->receive(message);
    string result;
    message >> result;
    return result;
  }
  template <class T, size_t n>
  void add(zmqpp::message &message, Tensor<T, n> &a) {
    TensorRM<T, n> temp;
    assign(temp, a);
    message.add_raw((const char *)temp.data(), temp.size() * sizeof(T));
  }
  template <class T, size_t n>
  string eval(string s, Tensor<T, n> &a) {
    if (mode < 0)
      return "";
    else if (mode < 1)
      THROW("uninitialized");
    string cmd;
    zmqpp::message message;
    message << cmd + s;
    add(message, a);
    socket->send(message);
    socket->receive(message);
    string response;
    message >> response;
    return response;
  }
  template <class T, size_t n, class S, size_t m>
  string eval(string s, Tensor<T, n> &a, Tensor<S, m> &b) {
    if (mode < 0)
      return "";
    else if (mode < 1)
      THROW("uninitialized");
    string cmd;
    zmqpp::message message;
    message << cmd + s;
    add(message, a);
    add(message, b);
    socket->send(message);
    socket->receive(message);
    string response;
    message >> response;
    return response;
  }
  string evalf(const char *format, ...) {
    static char buf[4096];
    va_list v;
    va_start(v, format);
    vsnprintf(buf, sizeof(buf), format, v);
    va_end(v);
    return eval(buf);
  }
  void clf() { eval("clf()"); }
  void subplot(int rows, int cols, int n) {
    eval(stringf("subplot(%d,%d,%d)", rows, cols, n));
  }
  void plot(Tensor<float, 1> &v, string extra = "") {
    if (extra != "") extra = string(",") + extra;
    if (v.rank() != 1) THROW("bad rank");
    eval(stringf("plot(farg(1)%s)", extra.c_str()), v);
  }
  void plot2(Tensor<float, 1> &u, Tensor<float, 1> &v, string extra = "") {
    if (extra != "") extra = string(",") + extra;
    if (u.rank() != 1) THROW("bad rank");
    if (v.rank() != 1) THROW("bad rank");
    eval(stringf("plot(farg(1),farg(2)%s)", extra.c_str()), u, v);
  }
  void imshow(Tensor<float, 2> &a, string extra = "") {
    if (extra != "") extra = string(",") + extra;
    eval(stringf("imshow(farg2(1,%d,%d)%s)", a.dimension(0), a.dimension(1),
                 extra.c_str()),
         a);
  }
  void imshowT(Tensor<float, 2> &a, string extra = "") {
    if (extra != "") extra = string(",") + extra;
    eval(stringf("imshow(farg2(1,%d,%d).T%s)", a.dimension(0), a.dimension(1),
                 extra.c_str()),
         a);
  }
};
#endif

inline PyServer *make_PyServer() { return new PyServer(); }
}

#endif
