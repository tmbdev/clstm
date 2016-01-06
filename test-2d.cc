#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <vector>
#include "clstm.h"
#include "clstmhl.h"
#include "extras.h"
#include "pstring.h"
#include "utils.h"

using namespace Eigen;
using namespace ocropus;
using std::vector;
using std::map;
using std::make_pair;
using std::shared_ptr;
using std::unique_ptr;
using std::cout;
using std::ifstream;
using std::set;
using std::to_string;
using std_string = std::string;
using std_wstring = std::wstring;
using std::regex;
using std::regex_replace;
#define string std_string
#define wstring std_wstring

double state = getdenv("seed", 17.9348);

inline double randu() {
  state = 189843.9384938 * state + 0.328340981343;
  state -= floor(state);
  return state;
}

inline double randn() {
  double u1 = randu();
  double u2 = randu();
  double r = -2 * log(u1);
  double theta = 2 * M_PI * u2;
  double z0 = r * cos(theta);
  return z0;
}

Float maxerr(Sequence &xs, Sequence &ys) {
  Float merr = 0.0;
  for (int t = 0; t < xs.size(); t++) {
    for (int i = 0; i < xs.rows(); i++) {
      for (int j = 0; j < ys.cols(); j++) {
        Float err = fabs(xs[t].v(i, j) - ys[t].v(i, j));
        merr = fmax(err, merr);
      }
    }
  }
  return merr;
}

Float meanerr(Sequence &xs, Sequence &ys) {
  Float merr = 0.0;
  Float count = 0.0;
  for (int t = 0; t < xs.size(); t++) {
    for (int i = 0; i < xs.rows(); i++) {
      for (int j = 0; j < ys.cols(); j++) {
        Float err = fabs(xs[t].v(i, j) - ys[t].v(i, j));
        merr += err;
        count += 1;
      }
    }
  }
  return merr / count;
}

void set_image(Sequence &seq, TensorMap3 image) {
  // image: (h, w, d) sequence: (t, d, h)
  int h = image.dimension(0);
  int w = image.dimension(1);
  int d = image.dimension(2);
  seq.resize(w, d, h);
  TensorMap4 t = seq.map4();
  seq.zero();
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
      for (int k = 0; k < d; k++) t(k, i, 0, j) = image(i, j, k);
}

void get_image(Tensor2 &image, Sequence &seq, int plane) {
  // image: (h, w, d) sequence: (t, d, h)
  int h = seq.cols();
  int w = seq.size();
  image.resize(h, w);
  TensorMap4 t = seq.map4();
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) image(i, j) = t(plane, i, 0, j);
}

void save_seq_as_image(const string &name, Sequence &seq, int plane = 0) {
  Tensor2 image;
  get_image(image, seq, plane);
  write_png(name.c_str(), image());
}

void gen_image(Sequence &input, Sequence &target) {
  int w = 256;
  int h = 256;
  EigenTensor3 image(w, h, 1);
  image.setZero();

  int r = 50;
  int x = int(randu() * (w - r));
  int y = int(randu() * (h - r));
  for (int i = 0; i < r; i++)
    for (int j = 0; j < r; j++) image(x + i, y + j, 0) = 1.0;

  set_image(target, image);
  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++) image(i, j, 0) += randn();

  set_image(input, image);
}

int main1(int argc, char **argv) {
  int ntrain = getienv("ntrain", 100000);
  Network net =
      make_net("twod", {{"ninput", 1}, {"nhidden", 3}, {"noutput", 1}});
  double lr = getdenv("lrate", 1e-4);
  net->setLearningRate(lr, 0.9);

  Sequence inputs, targets;
  for (int trial = 0; trial < ntrain; trial++) {
    gen_image(inputs, targets);
    set_inputs(net, inputs);
    net->forward();
    if (trial % 100 == 0)
      print(trial, maxerr(net->outputs, targets),
            meanerr(net->outputs, targets));
    set_targets(net, targets);
    net->backward();
    sgd_update(net);
    if (trial % 1000 == 0) {
      string base = "_";
      base += std::to_string(trial);
      save_seq_as_image(base + "_inputs.png", inputs);
      save_seq_as_image(base + "_outputs.png", net->outputs);
      save_seq_as_image(base + "_targets.png", targets);
      print("saved", base);
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  TRY { main1(argc, argv); }
  CATCH(const char *message) { cerr << "FATAL: " << message << endl; }
}
