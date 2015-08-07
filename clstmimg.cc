#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "multidim.h"
#include "pymulti.h"
#include "extras.h"

using std_string = std::string;
#define string std_string
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::to_string;
using std::make_pair;
using std::cout;
using std::stoi;
using namespace Eigen;
using namespace ocropus;
using namespace pymulti;

struct Image2Image : public ITrainable {
  mdarray<float> input, d_input, output, d_output;
  int idepth = -1;
  int odepth = -1;
  Network net;
  void setLearningRate(Float lr, Float momentum) {
    net->setLearningRate(lr, momentum);
  }
};

inline int clip(int x, int hi) {
  if (x > hi) return hi;
  if (x < 0) return 0;
  return x;
}

void batches_of_image_0(Sequence &batches, mdarray<float> &image, int depth,
                        int border = 0) {
  int d0 = image.dim(0), d1 = image.dim(1), d2 = image.dim(2);
  assert(depth == image.dim(2));
  int batch_depth = (2 * border + 1) * d2;
  batches.resize(d0);
  for (int i = 0; i < d0; i++) {
    batches[i].resize(batch_depth, d1);
    for (int j = 0; j < d1; j++) {
      int l = 0;
      for (int r = -border; r <= border; r++) {
        for (int k = 0; k < d2; k++)
          batches[i](l++, j) = image(clip(i + r, d0), j, k);
      }
      assert(l == batch_depth);
    }
  }
}

void image_of_batches_0(mdarray<float> &image, Sequence &batches, int depth,
                        int border = 0) {
  int d0 = batches.size();
  int d1 = COLS(batches[0]);
  int d2 = depth;
  assert((2 * border + 1) * depth == ROWS(batches[0]));
  image.resize(d0, d1, d2);
  for (int i = 0; i < d0; i++) {
    for (int j = 0; j < d1; j++) {
      int l = 0;
      for (int r = -border; r <= border; r++) {
        for (int k = 0; k < d2; k++)
          if (i + r >= 0 && i + r < d0) image(i + r, j, k) = batches[i](l++, j);
      }
      assert(l == ROWS(batches[i]));
    }
  }
}

struct Vstrips : public Image2Image {
  int border = 0;
  void initialize() {
    this->odepth = iattr("noutput");
    this->idepth = iattr("ninput");
    net.reset(make_LSTM1());
    net->init(odepth, (2 * border + 1) * idepth);
  }
  void forward() {
    assert(input.rank() == 3);
    assert(input.dim(2) == idepth);
    batches_of_image_0(net->inputs, input, idepth, border);
    // print("inputs", net->inputs.size(), ROWS(net->inputs[0]),
    // COLS(net->inputs[0]), "/", net->ninput());
    net->forward();
    image_of_batches_0(output, net->outputs, odepth);
  }
  void backward() {
    assert(d_output.rank() == 3);
    assert(output.dim(2) == odepth);
    batches_of_image_0(net->d_outputs, d_output, odepth);
    net->backward();
    image_of_batches_0(d_input, net->d_inputs, idepth, border);
  }
  void update() { net->update(); }
};

int main1(int argc, char **argv) {
  try {
    mdarray<float> input, output;
    Sequence seq;
    read_png(input, argv[1]);
    print("input", input.dim(0), input.dim(1), input.dim(2));
    print("irange", input.min(), input.max());
    batches_of_image_0(seq, input, 3);
    print("seq", seq.size(), ROWS(seq[0]), COLS(seq[0]));
    image_of_batches_0(output, seq, 3);
    write_png(argv[2], input);
  } catch (const char *msg) {
    print("ERROR:", msg);
  }
}

int main2(int argc, char **argv) {
  try {
    shared_ptr<Vstrips> net;
    net.reset(new Vstrips());
    net->border = 0;
    net->init(3, 3);
    read_png(net->input, argv[1]);
    net->forward();
    write_png(argv[2], net->input);
  } catch (const char *msg) {
    print("ERROR:", msg);
  }
}

int main3(int argc, char **argv) {
  double lrate = getrenv("lrate", 1e-3);
  double momentum = getrenv("momentum", 0.9);
  vector<string> files;
  string line;
  std::ifstream stream(argv[1]);
  while (getline(stream, line)) files.push_back(line);
  try {
    shared_ptr<Vstrips> net;
    net.reset(new Vstrips());
    net->border = 0;
    net->init(3, 3);
    net->setLearningRate(lrate, momentum);
    for (int trial = 0; trial < 1000000; trial++) {
      mdarray<float> image, out;
      string file = files[trial % files.size()];
      print(trial, file);
      read_png(image, file.c_str());
      print("image", image.dim(0), image.dim(1), image.dim(2));
      net->input = image;
      print("input", net->input.dim(0), net->input.dim(1), net->input.dim(2));
      print("irange", net->input.min(), net->input.max());
      net->forward();
      print("output", net->output.dim(0), net->output.dim(1),
            net->output.dim(2));
      print("orange", net->output.min(), net->output.max());
      out = net->output;
      out.clip(0.0, 1.0);
      write_png("temp.png", out);
      net->d_output = image;
      net->d_output -= net->output;
      net->backward();
      net->update();
    }
  } catch (const char *msg) {
    print("ERROR:", msg);
  }
}

double sqr(double x) { return x * x; }
double norm(Sequence &seq) {
  double err = 0.0;
  int dim = ROWS(seq[0]), bs = COLS(seq[0]);
  for (int t = 0; t < seq.size(); t++)
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < bs; j++) err += sqr(seq[t](i, j));
  return err;
}
void threshold(Sequence &seq, double threshold) {
  int dim = ROWS(seq[0]);
  int bs = COLS(seq[0]);
  for (int t = 0; t < seq.size(); t++)
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < bs; j++) seq[t](i, j) = (seq[t](i, j) > threshold);
}

int main4(int argc, char **argv) {
  double lrate = getrenv("lrate", 1e-4);
  double momentum = getrenv("momentum", 0.9);
  try {
    Network net;
    net.reset(make_LSTM1());
    net->init(2, 10, 2);
    net->setLearningRate(lrate, momentum);
    for (int trial = 0; trial < 1000000; trial++) {
      Sequence seq;
      seq.resize(20);
      for (int t = 0; t < seq.size(); t++) seq[t] = Mat::Random(2, 1);
      threshold(seq, 0.5);
      set_inputs(net.get(), seq);
      net->forward();
      set_targets(net.get(), seq);
      if (trial % 100 == 0) print(trial, norm(net->d_outputs));
      net->backward();
      net->update();
      if (trial % 10000 == 0) {
        Sequence &out = net->outputs;
        for (int t = 0; t < seq.size(); t++)
          print(t, ":", seq[t](0, 0), seq[t](1, 0), " / ", out[t](0, 0),
                out[t](1, 0));
      }
    }
  } catch (const char *msg) {
    print("ERROR:", msg);
  }
}

int main5(int argc, char **argv) {
  double lrate = getrenv("lrate", 1e-3);
  double momentum = getrenv("momentum", 0.9);
  try {
    Network lstm;
    lstm.reset(make_LSTM1());
    lstm->init(2, 2, 2);
    lstm->setLearningRate(lrate, momentum);
    shared_ptr<Vstrips> net;
    net.reset(new Vstrips());
    net->border = 0;
    net->net = lstm;
    net->idepth = 2;
    net->odepth = 2;
    for (int trial = 0; trial < 1000000; trial++) {
      mdarray<float> input, output;
      Sequence seq;
      seq.resize(20);
      for (int t = 0; t < seq.size(); t++)
        seq[t] = Mat::Random(2, 1).cwiseAbs();
      threshold(seq, 0.5);
      input.resize((int)seq.size(), (int)COLS(seq[0]), (int)ROWS(seq[0]));
      for (int t = 0; t < input.dim(0); t++)
        for (int i = 0; i < input.dim(1); i++)
          for (int j = 0; j < input.dim(2); j++) input(t, i, j) = seq[t](j, i);
      net->input = input;
      net->forward();
      output = net->output;
      net->d_output = input;
      net->d_output -= output;
      if (trial % 10000 == 0) {
        print("in", input.min(), input.max(), "out", output.min(), output.max(),
              "err", net->d_output.norm());
        for (int t = 0; t < input.dim(0); t++)
          print(t, ":", input(t, 0, 0), input(t, 0, 1), " / ", output(t, 0, 0),
                output(t, 0, 1));
      }
      net->backward();
      net->update();
    }
  } catch (const char *msg) {
    print("ERROR:", msg);
  }
}

#ifndef MAIN
#define MAIN main4
#endif

int main(int argc, char **argv) { return MAIN(argc, argv); }
