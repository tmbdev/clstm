#include "pstring.h"
#include "clstm.h"
#include "clstmhl.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <iostream>
#include <set>

#include "multidim.h"
#include "pymulti.h"
#include "extras.h"

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
using std::ofstream;
using std::set;
using std::to_string;
using std_string = std::string;
using std_wstring = std::wstring;
#define string std_string
#define wstring std_wstring

inline float scaled_log(float x) {
  const float thresh = 10.0;
  if (x <= 0.0) return 0.0;
  float l = log(x);
  if (l < -thresh) return 0.0;
  if (l > 0) return 1.0;
  return (l + thresh) / thresh;
}

void write_text(const string fname, const wstring &data) {
  string utf8 = utf32_to_utf8(data);
  ofstream stream(fname);
  stream << utf8 << endl;
}

void write_text(const string fname, const string &data) {
  ofstream stream(fname);
  stream << data << endl;
}

int main1(int argc, char **argv) {
  if (argc != 2) THROW("give text file as an argument");
  const char *fname = argv[1];

  string load_name = getsenv("load", "");
  if (load_name == "") THROW("must give load= parameter");
  CLSTMOCR clstm;
  clstm.load(load_name);

  bool conf = getienv("conf", 0);
  string output = getsenv("output", "text");
  bool save_text = getienv("save_text", 1);

  ifstream stream(fname);
  string line;
  while (getline(stream, line)) {
    mdarray<float> raw;
    string fname = line;
    string basename = fname.substr(0, fname.find_last_of("."));
    read_png(raw, fname.c_str(), true);
    for (int i = 0; i < raw.size(); i++) raw[i] = 1 - raw[i];
    if (!conf) {
      string out = clstm.predict_utf8(raw);
      cout << line << "\t" << out << endl;
      if (save_text) {
        write_text(basename+".txt", out);
      }
    } else {
      cout << "file " << line << endl;
      vector<CharPrediction> preds;
      clstm.predict(preds, raw);
      for (int i = 0; i < preds.size(); i++) {
        CharPrediction p = preds[i];
        const char *sep = "\t";
        cout << p.i << sep << p.x << sep << p.c << sep << p.p << endl;
      }
    }
    if (output == "text" ) {
      // nothing else to do
    } else if (output == "logs") {
      mdarray<float> outputs;
      clstm.get_outputs(outputs);
      for (int t=0; t<outputs.dim(0); t++)
        for (int c=0; c<outputs.dim(1); c++)
          outputs(t,c) = scaled_log(outputs(t,c));
      write_png((basename+".lp.png").c_str(), outputs);
    } else if (output == "posteriors") {
      mdarray<float> outputs;
      clstm.get_outputs(outputs);
      write_png((basename+".p.png").c_str(), outputs);
    } else {
      THROW("unknown output format");
    }
  }
  return 0;
}

int main(int argc, char **argv) {
#ifdef NOEXCEPTION
  return main1(argc, argv);
#else
  try {
    return main1(argc, argv);
  } catch (const char *message) {
    cerr << "FATAL: " << message << endl;
  }
#endif
}
