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
#include <Eigen/Dense>
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
map<string, INetworkFactory> network_factories;

int register_stacked(const string &name, function<void(INetwork *)> f) {
  network_factories[name] = [f]() {
    INetwork *stacked = make_Stacked();
    stacked->initializer = f;
    return stacked;
  };
  return 0;
}

#define C(X, Y) X##Y
#define REGISTER_STACKED(X) \
  int C(status_, X) = register_stacked(#X, C(init_, X));

void init_LSTM1(INetwork *net) {
  net->name = "lstm1";
  net->attributes["kind"] = "lstm1";
  int ni = net->irequire("ninput");
  int nh = net->irequire("nhidden");
  int no = net->irequire("noutput");
  Network fwd, softmax;
  fwd.reset(make_LSTM());
  fwd->init(nh, ni);
  net->add(fwd);
  softmax.reset(make_SoftmaxLayer());
  softmax->init(no, nh);
  net->add(softmax);
}
REGISTER_STACKED(LSTM1);

void init_REVLSTM1(INetwork *net) {
  net->name = "revlstm1";
  int ni = net->irequire("ninput");
  int nh = net->irequire("nhidden");
  int no = net->irequire("noutput");
  Network fwd, rev, softmax;
  fwd.reset(make_LSTM());
  fwd->init(nh, ni);
  rev.reset(make_Reversed());
  rev->add(fwd);
  net->add(rev);
  softmax.reset(make_SoftmaxLayer());
  softmax->init(no, nh);
  net->add(softmax);
}
REGISTER_STACKED(REVLSTM1);

#if 0
void init_BIDILSTM(INetwork *net) {
    net->name = "bidilstm";
    net->attributes["kind"] = "bidi";
    int ni = net->irequire("ninput");
    int nh = net->irequire("nhidden");
    int no = net->irequire("noutput");
    Network fwd, bwd, parallel, reversed, softmax;
    fwd = make_net(net->attr("lstm_type", "LSTM"));
    bwd = make_net(net->attr("lstm_type", "LSTM"));
    softmax = make_net(net->attr("output_type", "SoftmaxLayer"));
    fwd->init(nh, ni);
    bwd->init(nh, ni);
    reversed.reset(make_Reversed());
    reversed->add(bwd);
    parallel.reset(make_Parallel());
    parallel->add(fwd);
    parallel->add(reversed);
    net->add(parallel);
    softmax->init(no, 2*nh);
    net->add(softmax);
}
REGISTER_STACKED(BIDILSTM);

void init_LRBIDILSTM(INetwork *net) {
    net->set("output_type", "SigmoidLayer");
    init_BIDILSTM(net);
}
REGISTER_STACKED(LRBIDILSTM);

void init_LINBIDILSTM(INetwork *net) {
    net->set("output_type", "LinearLayer");
    init_BIDILSTM(net);
}
REGISTER_STACKED(LINBIDILSTM);

void init_BIDILSTM2(INetwork *net) {
    net->name = "bidilstm2";
    net->attributes["kind"] = "bidi2";
    int ni = net->irequire("ninput");
    int nh = net->irequire("nhidden");
    int nh2 = net->irequire("nhidden2");
    int no = net->irequire("noutput");
    // cerr << ">>> " << ni << " " << nh << " " << nh2 << " " << no << endl;
    Network parallel1, parallel2, logreg;
    parallel1.reset(make_BidiLayer());
    parallel1->init(nh, ni);
    net->add(parallel1);
    parallel2.reset(make_BidiLayer());
    parallel2->init(nh2, 2*nh);
    net->add(parallel2);
    logreg.reset(make_SoftmaxLayer());
    logreg->init(no, 2*nh2);
    net->add(logreg);
};
REGISTER_STACKED(BIDILSTM2);
#endif

static string fixup_net_name(string kind) {
  if (kind == "bidi") return "BIDILSTM";
  if (kind == "lrbidi") return "LRBIDILSTM";
  if (kind == "bidi2") return "BIDILSTM2";
  if (kind == "lstm1") return "LSTM1";
  if (kind == "uni") return "LSTM1";
  return kind;
}

Network make_net(string kind) {
  if (kind == "") THROW("empty network kind in make_net");
  Network net;
  net = make_layer(kind);
  if (net) return net;
  kind = fixup_net_name(kind);
  auto it = network_factories.find(kind);
  if (it == network_factories.end()) THROW("unknown network kind");
  net.reset(it->second());
  net->attributes["kind"] = kind;
  return net;
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

Network load_net_hdf5(const string &fname) {
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

void save_net_hdf5(const string &fname, Network net) {
  rename(fname.c_str(), (fname + "~").c_str());
  save_net_raw(fname.c_str(), net.get());
}
}

using namespace ocropus;
using namespace std;

int main(int argc, char **argv) {
  Network net = load_net_hdf5(argv[1]);
  save_as_proto(argv[2], net.get());
}
