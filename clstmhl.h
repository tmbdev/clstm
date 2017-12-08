// -*- C++ -*-

#ifndef ocropus_clstmhl_
#define ocropus_clstmhl_

#include <memory>
#include <string>
#include <vector>
#include "clstm.h"
#include "extras.h"
#include "pstring.h"
#include "tensor.h"

namespace ocropus {

struct CharPrediction {
  int i;
  int x;
  wchar_t c;
  float p;
};

// Clstm network used for text input and output.
struct CLSTMText {
  Network net;
  int nclasses = -1;
  int iclasses = -1;
  int neps = 3;
  Sequence targets;
  Sequence aligned;
  void setLearningRate(float lr, float mom) { net->setLearningRate(lr, mom); }

  // Loads the network from the given file. If the file does not exist
  // or the contents of the file cannot be read, throws an exception.
  void load(const std::string &fname) {
    if (!maybe_load(fname)) {
      THROW("Could not load CLSTMText net from file: " + fname);
    }
  }

  // Tries to load a network from the given file. If the file does not exist
  // or the contents of the file cannot be read, returns false.
  bool maybe_load(const std::string &fname) {
    net = maybe_load_net(fname);

    if (!net) {
      cerr << "WARNING: could not load CLSTMText net from " << fname;
      return false;
    }
    nclasses = net->codec.size();
    iclasses = net->icodec.size();
    int neps = net->attr.get("neps", -1);
    if (neps < 0) cerr << "WARNING: no neps\n";
    return true;
  }

  // Saves the network to the given file. If this operation fails, throws an
  // exception.
  void save(const std::string &fname) {
    if (!maybe_save(fname)) {
      THROW("Could not save CLSTMText net to file: " + fname);
    }
  }

  // Saves the network to the given file. If this operation fails, return false.
  bool maybe_save(const std::string &fname) {
    return maybe_save_net(fname, net);
  }

  void createBidi(const std::vector<int> &icodec, const std::vector<int> codec,
                  int nhidden) {
    // This is just the simplest case of creating a network. For more complex
    // networks, create them outside and assign them to "net".
    iclasses = icodec.size();
    nclasses = codec.size();
    net = make_net("bidi", {{"ninput", (int)icodec.size()},
                            {"noutput", (int)codec.size()},
                            {"nhidden", nhidden}});
    net->attr.set("neps", neps);
    net->icodec.set(icodec);
    net->codec.set(codec);
  }
  void setInputs(const std::wstring &s) {
    Classes cs;
    net->icodec.encode(cs, s);
    Sequence &seq = net->inputs;
    int d = net->ninput();
    seq.clear();
    seq.resize(cs.size() * (neps + 1) + neps, d, 1);
    int index = 0;
    for (int i = 0; i < neps; i++) seq[index++].clear();
    for (int pos = 0; pos < cs.size(); pos++) {
      TensorMap2 v = *seq[index++].v;
      v.setZero();
      v(cs[pos], 0) = 1.0;
      for (int i = 0; i < neps; i++) seq[index++].v.setZero();
    }
    assert(index == seq.size());
    seq.check();
  }

  // Trains the network using the given input and target using backpropagation.
  std::wstring train(const std::wstring &in, const std::wstring &target) {
    setInputs(in);
    net->forward();
    Classes transcript;
    net->codec.encode(transcript, target);
    mktargets(targets, transcript, nclasses);
    ctc_align_targets(aligned, net->outputs, targets);
    for (int t = 0; t < aligned.size(); t++)
      net->outputs[t].d() = aligned[t].v() - net->outputs[t].v();
    net->backward();
    sgd_update(net);
    Classes output_classes;
    trivial_decode(output_classes, net->outputs);
    return net->codec.decode(output_classes);
  }
  std::wstring predict(const std::wstring &in) {
    setInputs(in);
    net->forward();
    Classes output_classes;
    trivial_decode(output_classes, net->outputs);
    return net->codec.decode(output_classes);
  }
  void train_utf8(const std::string &in, const std::string &target) {
    train(utf8_to_utf32(in), utf8_to_utf32(target));
  }
  std::string aligned_utf8() {
    Classes outputs;
    trivial_decode(outputs, aligned);
    std::wstring temp = net->codec.decode(outputs);
    return utf32_to_utf8(temp);
  }
  std::string predict_utf8(const std::string &in) {
    return utf32_to_utf8(predict(utf8_to_utf32(in)));
  }
  void get_outputs(Tensor2 &outputs) {
    Sequence &o = net->outputs;
    outputs.resize(int(o.size()), int(o[0].rows()));
    for (int t = 0; t < outputs.dimension(0); t++)
      for (int c = 0; c < outputs.dimension(1); c++)
        outputs(t, c) = net->outputs[t].v(c, 0);
  }
};

struct CLSTMOCR {
  shared_ptr<INormalizer> normalizer;
  Network net;
  int target_height = 48;
  int nclasses = -1;
  Sequence aligned, targets;
  Tensor2 image;
  void setLearningRate(float lr, float mom) { net->setLearningRate(lr, mom); }

  // Tries to load a network from the given file. If the file does not exist
  // or the contents of the file cannot be read, returns false.
  bool maybe_load(const std::string &fname) {
    net = maybe_load_net(fname);
    if (!net) {
      cerr << "WARNING: could not load CLSTMOCR net from " << fname;
      return false;
    }
    nclasses = net->codec.size();
    target_height = net->ninput();
    normalizer.reset(make_CenterNormalizer());
    normalizer->target_height = target_height;
    return true;
  }

  // Loads the network from the given file. If the file does not exist
  // or the contents of the file cannot be read, throws an exception.
  void load(const std::string &fname) {
    if (!maybe_load(fname)) {
      THROW("Could not load CLSTMOCR net from file: " + fname);
    }
  }

  // Saves the network to the given file. If this operation fails, throws an
  // exception.
  void save(const std::string &fname) {
    if (!maybe_save(fname)) {
      THROW("Could not save CLSTMOCR net to file: " + fname);
    }
  }

  // Saves the network to the given file. If this operation fails, return false.
  bool maybe_save(const std::string &fname) {
    return maybe_save_net(fname, net);
  }

  void createBidi(const std::vector<int> codec, int nhidden) {
    nclasses = codec.size();
    net = make_net("bidi", {{"ninput", target_height},
                            {"noutput", nclasses},
                            {"nhidden", nhidden}});
    net->initialize();
    net->codec.set(codec);
    normalizer.reset(make_CenterNormalizer());
    normalizer->target_height = target_height;
  }
  std::wstring fwdbwd(TensorMap2 raw, const std::wstring &target) {
    normalizer->measure(raw);
    image.like(raw);
    normalizer->normalize(image, raw);
    set_inputs(net, image());
    net->forward();
    Classes transcript;
    net->codec.encode(transcript, target);
    mktargets(targets, transcript, nclasses);
    ctc_align_targets(aligned, net->outputs, targets);
    for (int t = 0; t < aligned.size(); t++)
      net->outputs[t].d() = aligned[t].v() - net->outputs[t].v();
    net->backward();
    Classes outputs;
    trivial_decode(outputs, net->outputs);
    return net->codec.decode(outputs);
  }
  void update() { sgd_update(net); }
  std::wstring train(TensorMap2 raw, const std::wstring &target) {
    std::wstring result = fwdbwd(raw, target);
    update();
    return result;
  }
  std::string aligned_utf8() {
    Classes outputs;
    trivial_decode(outputs, aligned);
    std::wstring temp = net->codec.decode(outputs);
    return utf32_to_utf8(temp);
  }
  std::string train_utf8(TensorMap2 raw, const std::string &target) {
    return utf32_to_utf8(train(raw, utf8_to_utf32(target)));
  }
  std::wstring predict(TensorMap2 raw, vector<int> *where = 0) {
    normalizer->measure(raw);
    image.like(raw);
    normalizer->normalize(image, raw);
    set_inputs(net, image());
    net->forward();
    Classes outputs;
    trivial_decode(outputs, net->outputs, 0, where);
    return net->codec.decode(outputs);
  }
  void predict(vector<CharPrediction> &preds, TensorMap2 raw) {
    normalizer->measure(raw);
    image.like(raw);
    normalizer->normalize(image, raw);
    set_inputs(net, image());
    net->forward();
    Classes outputs;
    vector<int> where;
    trivial_decode(outputs, net->outputs, 0, &where);
    preds.clear();
    for (int i = 0; i < outputs.size(); i++) {
      int t = where[i];
      int cls = outputs[i];
      wchar_t c = net->codec.decode(outputs[i]);
      float p = net->outputs[t].v(cls, 0);
      CharPrediction pred{i, t, c, p};
      preds.push_back(pred);
    }
  }
  std::string predict_utf8(TensorMap2 raw) {
    return utf32_to_utf8(predict(raw));
  }
  void get_outputs(Tensor2 &outputs) {
    Sequence &o = net->outputs;
    outputs.resize(int(o.size()), int(o[0].rows()));
    for (int t = 0; t < outputs.dimension(0); t++)
      for (int c = 0; c < outputs.dimension(1); c++)
        outputs(t, c) = net->outputs[t].v(c, 0);
  }
};
}

#endif
