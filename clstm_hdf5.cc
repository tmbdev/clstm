// Loading and saving networks with HDF5. This is obsolete, use the
// protobuf based implementation instead.

#include "clstm.h"
#include "h5eigen.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include "clstm.pb.h"
#include <stdarg.h>

namespace {
inline void print() { std::cout << std::endl; }

template <class T>
inline void print(const T &arg) {
  std::cout << arg << std::endl;
}

template <class T, typename... Args>
inline void print(T arg, Args... args) {
  std::cout << arg << " ";
  print(args...);
}
}

namespace ocropus {
using std::cout;
using std::endl;

void save_codec(h5eigen::HDF5 *h5, const char *name, vector<int> &codec,
                int n) {
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
    h5->attr.set(kv.first, kv.second);
  }
  save_codec(h5.get(), "codec", net->codec, net->noutput());
  save_codec(h5.get(), "icodec", net->icodec, net->ninput());

  net->weights("", [&h5](const string &prefix, VecMat a, VecMat da) {
    if (a.mat)
      h5->put(*a.mat, prefix.c_str());
    else if (a.vec)
      h5->put(*a.vec, prefix.c_str());
    else
      THROW("oops (save type)");
  });
}

void load_net_raw(INetwork *net, const char *fname) {
  using namespace h5eigen;
  unique_ptr<HDF5> h5(make_HDF5());
  h5->open(fname);
  h5->attr.gets(net->attributes);
  load_codec(net->icodec, h5.get(), "icodec");
  load_codec(net->codec, h5.get(), "codec");
  net->weights("", [&h5](const string &prefix, VecMat a, VecMat da) {
    if (a.mat)
      h5->get(*a.mat, prefix.c_str());
    else if (a.vec)
      h5->get1d(*a.vec, prefix.c_str());
    else
      THROW("oops (load type)");
  });
  net->networks("", [](string s, INetwork *net) { net->postLoad(); });
}

void load_attributes(map<string, string> &attributes, const string &fname) {
  using namespace h5eigen;
  unique_ptr<HDF5> h5(make_HDF5());
  h5->open(fname.c_str());
  h5->attr.gets(attributes);
}

Network load_net(const string &fname) {
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
  Network net;
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

void save_net(const string &fname, Network net) {
  rename(fname.c_str(), (fname + "~").c_str());
  save_net_raw(fname.c_str(), net.get());
}
}
