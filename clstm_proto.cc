// Loading and saving networks using protobuf library.
// See clstm.proto for the protocol buffer definitions used here.

#include "clstm.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>
#include "utils.h"
#ifdef GOOGLE
#include "third_party/clstm/tensor/clstm.pb.h"
#else
#include "clstm.pb.h"
#endif

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

void proto_of_params(clstm::Array *array, Params &params, bool weights = true) {
  Tensor2 temp;
  temp = params.v;  // copy values in case they are on GPU
  TensorMap2 a = temp();
  int n = a.dimension(0), m = a.dimension(1);
  array->add_dim(n);
  array->add_dim(m);
  if (!weights) return;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) array->add_value(a(i, j));
}

void params_of_proto(Params &params, const clstm::Array *array) {
  if (array->dim_size() != 2)
    throwf("bad format (Mat, %s, %d)", array->name().c_str(),
           array->dim_size());
  params.setZero(array->dim(0), array->dim(1));
  TensorMap2 a = params.v();
  if (array->value_size() > 0) {
    if (array->value_size() != a.size()) THROW("bad size (Mat)");
    int k = 0;
    for (int i = 0; i < a.dimension(0); i++)
      for (int j = 0; j < a.dimension(1); j++) a(i, j) = array->value(k++);
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
    proto_of_params(array, *a, weights);
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
    net->attr.set(attr->key(), std::string(attr->value()));
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
    params_of_proto(*a, &proto->weights(i));
  }
  for (int i = 0; i < proto->sub_size(); i++) {
    net->add(net_of_proto(&proto->sub(i)));
    net->sub[i]->attr.super = &net->attr;
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

bool write_as_proto(ostream &output, INetwork *net) {
  unique_ptr<clstm::NetworkProto> proto;
  proto.reset(new clstm::NetworkProto());
  proto_of_net(proto.get(), net);
  return proto->SerializeToOstream(&output);
}

bool save_as_proto(const string &fname, INetwork *net) {
  ofstream stream;
  stream.open(fname, ios::binary);
  return write_as_proto(stream, net);
}

Network read_as_proto(istream &stream) {
  unique_ptr<clstm::NetworkProto> proto;
  proto.reset(new clstm::NetworkProto());
  if (proto->ParseFromIstream(&stream) == false) {
    return Network();
  }
  return net_of_proto(proto.get());
}

Network load_as_proto(const string &fname) {
  ifstream stream;
  stream.open(fname, ios::binary);
  if (!stream) throwf("cannot open: %s", fname.c_str());
  return read_as_proto(stream);
}
}
