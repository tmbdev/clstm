// Loading and saving networks using protobuf library.
// See clstm.proto for the protocol buffer definitions used here.

#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <stdarg.h>
#include <typeinfo>
#ifdef GOOGLE
#include "third_party/clstm/clstm.pb.h"
#else
#include "clstm.pb.h"
#endif

namespace {
inline void throwf(const char *format, ...) {
  static char buf[1024];
  va_list arglist;
  va_start(arglist, format);
  vsprintf(buf, format, arglist);
  va_end(arglist);
  THROW(buf);
}

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
using std::ostream;
using std::istream;
using std::ios;
using std::ofstream;
using std::ifstream;
using std::to_string;

bool proto_verbose =
    getenv("clstm_proto_verbose") && atoi(getenv("clstm_proto_verbose"));

void proto_of_Mat(clstm::Array *array, Mat &a, bool weights = true) {
  array->add_dim(a.rows());
  array->add_dim(a.cols());
  if (!weights) return;
  for (int i = 0; i < a.rows(); i++)
    for (int j = 0; j < a.cols(); j++) array->add_value(a(i, j));
}

void Mat_of_proto(Mat &a, const clstm::Array *array) {
  if (array->dim_size() != 2)
    throwf("bad format (Mat, %s, %d)", array->name().c_str(),
           array->dim_size());
  a.resize(array->dim(0), array->dim(1));
  a.setZero();
  if (array->value_size() > 0) {
    if (array->value_size() != a.size()) THROW("bad size (Mat)");
    int k = 0;
    for (int i = 0; i < a.rows(); i++)
      for (int j = 0; j < a.cols(); j++) a(i, j) = array->value(k++);
  }
}

void proto_of_net(clstm::NetworkProto *proto, INetwork *net,
                  bool weights = true) {
  if (net->kind == "") {
    cerr << typeid(*net).name() << endl;
    assert(net->kind != "");
  }
  proto->set_kind(net->kind);
  proto->set_ninput(net->ninput());
  proto->set_noutput(net->noutput());
  assert(proto->kind() != "");
  assert(proto->ninput() >= 0);
  assert(proto->ninput() < 1000000);
  assert(proto->noutput() >= 0);
  assert(proto->noutput() < 1000000);
  for (int i = 0; i < net->icodec.size(); i++)
    proto->add_icodec(net->icodec.codec[i]);
  for (int i = 0; i < net->codec.size(); i++)
    proto->add_codec(net->codec.codec[i]);
  for (auto kv : net->attr) {
    if (kv.first == "name") continue;
    if (kv.first == "ninput") continue;
    if (kv.first == "noutput") continue;
    clstm::KeyValue *kvp = proto->add_attribute();
    kvp->set_key(kv.first);
    kvp->set_value(kv.second);
  }
  for (auto it : net->parameters) {
    Params *a = it.second;
    string name = it.first;
    clstm::Array *array = proto->add_weights();
    array->set_name(name);
    proto_of_Mat(array, a->v, weights);
  }
  for (int i = 0; i < net->sub.size(); i++) {
    clstm::NetworkProto *subproto = proto->add_sub();
    proto_of_net(subproto, net->sub[i].get(), weights);
  }
}

Network net_of_proto(const clstm::NetworkProto *proto) {
  Network net;
  assert(proto->kind() != "");
  assert(proto->ninput() >= 0);
  assert(proto->ninput() < 1000000);
  assert(proto->noutput() >= 0);
  assert(proto->noutput() < 1000000);
  net = make_layer(proto->kind());
  net->attr.set("ninput", proto->ninput());
  net->attr.set("noutput", proto->noutput());
  for (int i = 0; i < proto->attribute_size(); i++) {
    const clstm::KeyValue *attr = &proto->attribute(i);
    net->attr.set(attr->key(), attr->value());
  }
  vector<int> icodec;
  for (int i = 0; i < proto->icodec_size(); i++)
    icodec.push_back(proto->icodec(i));
  net->icodec.set(icodec);
  vector<int> codec;
  for (int i = 0; i < proto->codec_size(); i++)
    codec.push_back(proto->codec(i));
  net->codec.set(codec);
  map<string, Params *> weights;
  for (auto it : net->parameters) {
    weights[it.first] = it.second;
  }
  for (int i = 0; i < proto->weights_size(); i++) {
    string key = proto->weights(i).name();
    Params *a = weights[key];
    Mat_of_proto(a->v, &proto->weights(i));
  }
  for (int i = 0; i < proto->sub_size(); i++) {
    net->add(net_of_proto(&proto->sub(i)));
  }
  net->postLoad();
  return net;
}

Network proto_clone_net(INetwork *net) {
  clstm::NetworkProto *proto = new clstm::NetworkProto();
  proto_of_net(proto, net);
  Network net2 = net_of_proto(proto);
  return net2;
}

void debug_as_proto(INetwork *net, bool weights) {
  clstm::NetworkProto *proto = new clstm::NetworkProto();
  proto_of_net(proto, net, weights);
  cout << proto->DebugString();
  delete proto;
}

void write_as_proto(ostream &output, INetwork *net) {
  unique_ptr<clstm::NetworkProto> proto;
  proto.reset(new clstm::NetworkProto());
  proto_of_net(proto.get(), net);
  if (proto->SerializeToOstream(&output) == false) {
    THROW("Serializing failed.");
  }
}

void save_as_proto(const string &fname, INetwork *net) {
  ofstream stream;
  stream.open(fname, ios::binary);
  write_as_proto(stream, net);
}

Network load_as_proto(const string &fname) {
  ifstream stream;
  stream.open(fname, ios::binary);
  if (!stream) throwf("%s: cannot open", fname.c_str());
  unique_ptr<clstm::NetworkProto> proto;
  proto.reset(new clstm::NetworkProto());
  if (proto->ParseFromIstream(&stream) == false) {
    THROW("Invalid message");
  }
  return net_of_proto(proto.get());
}
}
