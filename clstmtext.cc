#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>

#include "multidim.h"
#include "pymulti.h"
#include "extras.h"

#include "pstring.h"
#include "clstm.h"

using namespace Eigen;
using namespace ocropus;
using namespace pymulti;
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
#define string std_string
#define wstring std_wstring

struct Sample {
  wstring in, out;
};

void read_samples(vector<Sample> &samples, const string &fname) {
  ifstream stream(fname);
  string line;
  wstring in, out;
  ;
  samples.clear();
  while (getline(stream, line)) {
    // skip blank lines and lines starting with a comment
    if (line.substr(0, 2) == "# ") continue;
    if (line.size() == 0) continue;
    int where = line.find("\t");
    if (where < 0) THROW("no tab found in input line");
    in = utf8_to_utf32(line.substr(0, where));
    out = utf8_to_utf32(line.substr(where + 1));
    if (in.size() == 0) continue;
    if (out.size() == 0) continue;
    samples.push_back(Sample{in, out});
  }
}

void read_samples2(vector<Sample> &samples, const string &fname,
                   const string &fname2) {
  string line;
  samples.clear();
  wstring empty;
  {
    ifstream stream(fname);
    while (getline(stream, line)) {
      // skip blank lines and lines starting with a comment
      if (line.substr(0, 2) == "# ") continue;
      if (line.size() == 0) continue;
      wstring in = utf8_to_utf32(line);
      samples.push_back(Sample{in, empty});
    }
  }

  {
    ifstream stream(fname);
    int n = 0;
    while (getline(stream, line)) {
      // skip blank lines and lines starting with a comment
      if (line.substr(0, 2) == "# ") continue;
      if (line.size() == 0) continue;
      if (n >= samples.size()) THROW("too many lines in output file");
      wstring out = utf8_to_utf32(line);
      samples[n].out = out;
      n++;
    }
    if (n < samples.size()) THROW("too few lines in output file");
  }
}

void get_codec(vector<int> &codec, vector<Sample> &samples,
               wstring Sample::*p) {
  set<int> codes;
  codes.insert(0);
  for (auto e : samples) {
    for (auto c : e.*p) codes.insert(int(c));
  }
  for (auto c : codes) codec.push_back(c);
  for (int i = 1; i < codec.size(); i++) assert(codec[i] > codec[i - 1]);
}

void set_inputs_with_eps(INetwork *net, wstring &s, int neps) {
  Classes cs;
  net->iencode(cs, s);
  Sequence &seq = net->inputs;
  int d = net->ninput();
  seq.clear();
  seq.resize(cs.size() * (neps + 1) + neps);
  for (int i = 0; i < neps; i++) seq[i].setZero(d, 1);
  for (int pos = 0; pos < cs.size(); pos++) {
    seq[pos].setZero(d, 1);
    seq[pos](cs[pos], 0) = 1.0;
    for (int i = 0; i < neps; i++) seq[pos + 1 + i].setZero(d, 1);
  }
}

double error_rate(Network net, const string &testset, int nclasses, int neps) {
  int maxeval = getienv("maxeval", 1000000000);
  vector<Sample> samples;
  read_samples(samples, testset);

  int N = fmin(samples.size(), maxeval);
  double errs = 0.0;
  double total = 0;

  for (int sample = 0; sample < N; sample++) {
    Classes cs;
    set_inputs_with_eps(net.get(), samples[sample].in, neps);
    mdarray<float> image;
    assign(image, net->inputs);
    net->forward();
    mdarray<float> outputs;
    assign(outputs, net->outputs);
    Classes output_classes;
    trivial_decode(output_classes, net->outputs);
    wstring out = net->decode(output_classes);
    wstring gt = samples[sample].out;
    double err = levenshtein(gt, out);
    errs += err;
    total += gt.size();
  }
  return errs / total;
}

int main_train(int argc, char **argv) {
  srandomize();

  vector<Sample> samples;
  if (argc == 2) {
    read_samples(samples, argv[1]);
  } else if (argc == 3) {
    read_samples2(samples, argv[1], argv[2]);
  } else {
    THROW("args: training.txt [output.txt]");
  }
  print("got", samples.size(), "lines");
  int nsamples = samples.size();

  string load_name = getsenv("load", "");
  int save_every = getienv("save_every", 0);
  string save_name = getsenv("save_name", "");
  if (save_every >= 0 && save_name == "") THROW("must give save_name=");
  if (save_every > 0 && save_name.find('%') == string::npos)
    save_name += "-%08d.clstm";
  else
    save_name += ".clstm";
  string after_save = getsenv("after_save", "");

  int ntrain = getienv("ntrain", 1000000);
  double lrate = getrenv("lrate", 1e-4);
  int nhidden = getrenv("nhidden", getrenv("hidden", 100));
  int nhidden2 = getrenv("nhidden2", getrenv("hidden2", -1));
  int batch = getrenv("batch", 1);
  double momentum = getuenv("momentum", 0.9);
  int display_every = getienv("display_every", 0);
  int report_every = getienv("report_every", 1);
  bool randomize = getienv("randomize", 1);
  string lrnorm = getsenv("lrnorm", "batch");
  int neps = int(getuenv("neps", 3));
  string net_type = getsenv("lstm", "bidi");
  string lstm_type = getsenv("lstm_type", "NPLSTM");
  string output_type = getsenv("output_type", "SoftmaxLayer");

  string testset = getsenv("testset", "");
  int test_every = getienv("test_every", -1);
  string after_test = getsenv("after_test", "");

  print("params", "hg_version", HGVERSION, "lrate", lrate, "nhidden", nhidden,
        "nhidden2", nhidden2, "batch", batch, "momentum", momentum);

  unique_ptr<PyServer> py;
  if (display_every > 0) {
    py.reset(new PyServer());
    if (display_every > 0) py->open();
    py->eval("ion()");
    py->eval("matplotlib.rc('xtick',labelsize=7)");
    py->eval("matplotlib.rc('ytick',labelsize=7)");
    py->eval("matplotlib.rcParams.update({'font.size':7})");
  }

  Network net;
  int nclasses = -1, iclasses = -1;
  if (load_name != "") {
    net = load_net(load_name);
    nclasses = net->codec.size();
    iclasses = net->icodec.size();
    neps = net->attr.get("neps");
  } else {
    vector<int> icodec, codec;
    get_codec(icodec, samples, &Sample::in);
    get_codec(codec, samples, &Sample::out);
    iclasses = icodec.size();
    nclasses = codec.size();
    net = make_net(net_type, {{"ninput", iclasses},
                              {"noutput", nclasses},
                              {"nhidden", nhidden},
                              {"nhidden2", nhidden2},
                              {"lstm_type", lstm_type},
                              {"output_type", output_type},
                              {"neps", neps}});
    net->icodec = icodec;
    net->codec = codec;
  }
  net->setLearningRate(lrate, momentum);
  net->makeEncoders();
  net->info("");
  print("codec", net->codec.size(), "icodec", net->icodec.size());
  INetwork::Normalization norm = INetwork::NORM_DFLT;
  if (lrnorm == "len") norm = INetwork::NORM_LEN;
  if (lrnorm == "none") norm = INetwork::NORM_NONE;
  if (norm != INetwork::NORM_DFLT) print("nonstandard lrnorm: ", lrnorm);
  net->networks("",
                [norm](string s, INetwork *net) { net->normalization = norm; });

  Sequence targets;
  Sequence saligned;
  Classes classes;

  double start_time = now();
  double best_erate = 1e38;

  int start = net->attr.get("trial", getienv("start", -1)) + 1;
  if (start > 0) print("start", start);
  for (int trial = start; trial < ntrain; trial++) {
    bool report = (report_every > 0) && (trial % report_every == 0);
    int sample = trial % nsamples;
    if (randomize) sample = irandom() % nsamples;
    if (trial > 0 && save_every > 0 && trial % save_every == 0) {
      char fname[4096];
      sprintf(fname, save_name.c_str(), trial);
      print("saving", fname);
      net->attr.set("trial", trial);
      save_net(fname, net);
      if (after_save != "") system(after_save.c_str());
      cout.flush();
    }
    if (trial > 0 && test_every > 0 && trial % test_every == 0 &&
        testset != "") {
      double erate = error_rate(net, testset, nclasses, neps);
      print("TESTERR", now() - start_time, save_name, trial, erate, "lrate",
            lrate, "hidden", nhidden, nhidden2, "batch", batch, "momentum",
            momentum);
      if (save_every == 0 && erate < best_erate) {
        best_erate = erate;
        print("saving", save_name, "at", erate);
        net->attr.set("trial", trial);
        net->attr.set("last_err", best_erate);
        save_net(save_name, net);
        if (after_save != "") system(after_save.c_str());
      }
      if (after_test != "") system(after_test.c_str());
      cout.flush();
    }
    set_inputs_with_eps(net.get(), samples[sample].in, neps);
    mdarray<float> image;
    assign(image, net->inputs);
    Classes transcript;
    net->encode(transcript, samples[sample].out);
    net->forward();
    classes = transcript;
    mdarray<float> outputs;
    assign(outputs, net->outputs);
    mktargets(targets, classes, nclasses);
    ctc_align_targets(saligned, net->outputs, targets);
    assert(saligned.size() == net->outputs.size());
    for (int t = 0; t < saligned.size(); t++)
      net->outputs[t].d = saligned[t] - net->outputs[t];
    net->backward();
    if (trial % batch == 0) net->update();
    mdarray<float> aligned;
    assign(aligned, saligned);
    if (anynan(outputs) || anynan(aligned)) {
      print("got nan");
      break;
    }
    Classes output_classes, aligned_classes;
    trivial_decode(output_classes, net->outputs);
    trivial_decode(aligned_classes, saligned);
    wstring gt = net->decode(transcript);
    wstring out = net->decode(output_classes);
    wstring aln = net->decode(aligned_classes);
    if (report) {
      wstring s = samples[sample].in;
      print("trial", trial);
      print("INP:", "'" + utf32_to_utf8(s) + "'");
      print("TRU:", "'" + utf32_to_utf8(gt) + "'");
      print("OUT:", "'" + utf32_to_utf8(out) + "'");
      print("ALN:", "'" + utf32_to_utf8(aln) + "'");
      print(levenshtein(gt, out));
      cout.flush();
    }

    if (display_every > 0 && trial % display_every == 0) {
      py->eval("clf()");
      py->subplot(4, 1, 1);
      py->evalf("title('%s')", gt.c_str());
      py->imshowT(image, "cmap=cm.gray,interpolation='bilinear'");
      py->subplot(4, 1, 2);
      py->evalf("title('%s')", out.c_str());
      py->imshowT(outputs, "cmap=cm.hot,interpolation='bilinear'");
      py->subplot(4, 1, 3);
      py->evalf("title('%s')", aln.c_str());
      py->imshowT(aligned, "cmap=cm.hot,interpolation='bilinear'");
      py->subplot(4, 1, 4);
      mdarray<float> v;
      v.resize(outputs.dim(0));
      for (int t = 0; t < outputs.dim(0); t++) v(t) = outputs(t, 0);
      py->plot(v, "color='b'");
      int sp = 1;
      for (int t = 0; t < outputs.dim(0); t++) v(t) = outputs(t, sp);
      py->plot(v, "color='g'");
      int nclass = net->outputs[0].size();
      for (int t = 0; t < outputs.dim(0); t++)
        v(t) = net->outputs[t].col(0).segment(2, nclass - 2).maxCoeff();
      py->evalf("xlim(0,%d)", outputs.dim(0));
      py->plot(v, "color='r'");
      py->eval("ginput(1,1e-3)");
      cout.flush();
    }
  }
  return 0;
}

int main_filter(int argc, char **argv) {
  if (argc != 2) THROW("give text file as an argument");
  int display_every = getienv("display_every", -1);
  double display_delay = getdenv("display_delay", 1e-3);
  const char *fname = argv[1];
  string load_name = getsenv("load", "");
  if (load_name == "") THROW("must give load= parameter");
  Network net;
  net = load_net(load_name);
  int neps = net->attr.get("neps");
  dprint("codec", net->codec.size(), "icodec", net->icodec.size(), "neps",
         neps);

  unique_ptr<PyServer> py;
  if (display_every > 0) {
    py.reset(new PyServer());
    if (display_every > 0) py->open();
    py->eval("ion()");
    py->eval("matplotlib.rc('xtick',labelsize=7)");
    py->eval("matplotlib.rc('ytick',labelsize=7)");
    py->eval("matplotlib.rcParams.update({'font.size':7})");
  }

  string line;
  wstring in, out;

  ifstream stream(fname);
  int trial = 0;
  while (getline(stream, line)) {
    int where = line.find("\t");
    if (where >= 0) line = line.substr(0, where);
    in = utf8_to_utf32(line);
    set_inputs_with_eps(net.get(), in, neps);
    net->forward();
    Classes output_classes;
    trivial_decode(output_classes, net->outputs);
    wstring out = net->decode(output_classes);
    string out8 = utf32_to_utf8(out);
    print(out8);
    if (display_every > 0 && trial % display_every == 0) {
      mdarray<float> inputs, outputs;
      assign(inputs, net->inputs);
      assign(outputs, net->outputs);
      py->eval("clf()");
      py->subplot(2, 1, 1);
      py->eval("set_aspect('auto')");
      py->evalf("title(unicode('%s','utf-8'))", line.c_str());
      py->imshowT(inputs, "cmap=cm.gray,interpolation='bilinear'");
      py->subplot(2, 1, 2);
      py->eval("set_aspect('auto')");
      py->evalf("title(unicode('%s','utf-8'))", out.c_str());
      py->imshowT(outputs, "cmap=cm.gray,interpolation='bilinear'");
      py->evalf("ginput(1,%g)", display_delay);
    }
    trial++;
  }
  return 0;
}

const char *usage = /*program+*/
    "training.txt\n\n"
    "training.txt is a text file consisting of lines of the form:\n\n"
    "input\toutput\n\n"
    "UTF-8 encoding is assumed.\n";

int main(int argc, char **argv) {
  if (argc < 2) {
    print(string(argv[0]) + " " + usage);
    exit(1);
  }
  TRY {
    string mode = getsenv("mode", "train");
    if (mode == "train") {
      return main_train(argc, argv);
    } else if (mode == "filter") {
      return main_filter(argc, argv);
    }
  } CATCH (const char *msg) {
    print("EXCEPTION", msg);
  } CATCH (...) {
    print("UNKNOWN EXCEPTION");
  }
}
