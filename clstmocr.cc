#include "clstm.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>
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

int print_usage(char **argv) {
    cerr << "Usage: [VAR=VAL...] " << argv[0] << " IMAGEFILE\n";
    cerr << "\n";
    cerr << "  Arguments:\n";
    cerr << "    IMAGEFILE      Image file to OCR\n";
    cerr << "  \n";
    cerr << "  Variables:\n";
    cerr << "     load          Model to recognize with. Required\n";
    cerr << "     conf          Output character-wise predictions. Default: 0\n";
    cerr << "     output        Output format, either 'text' or 'posteriors'. Default: 'text'\n";
    cerr << "     save_text     Save text to IMAGEFILE.txt. Default: 1\n";
    return EXIT_FAILURE;
}

int main1(int argc, char **argv) {
  if (argc != 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))
      return print_usage(argv);
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
    Tensor2 raw;
    string fname = line;
    string basename = fname.substr(0, fname.find_last_of("."));
    read_png(raw, fname.c_str());
    raw() = -raw() + Float(1.0);
    if (!conf) {
      string out = clstm.predict_utf8(raw());
      cout << line << "\t" << out << endl;
      if (save_text) {
        write_text(basename + ".txt", out);
      }
    } else {
      cout << "file " << line << endl;
      vector<CharPrediction> preds;
      clstm.predict(preds, raw());
      for (int i = 0; i < preds.size(); i++) {
        CharPrediction p = preds[i];
        const char *sep = "\t";
        cout << p.i << sep << p.x << sep << p.c << sep << p.p << endl;
      }
    }
    if (output == "text") {
      // nothing else to do
    } else if (output == "logs") {
      Tensor2 outputs;
      clstm.get_outputs(outputs);
      for (int t = 0; t < outputs.dimension(0); t++)
        for (int c = 0; c < outputs.dimension(1); c++)
          outputs(t, c) = scaled_log(outputs(t, c));
      write_png((basename + ".lp.png").c_str(), outputs());
    } else if (output == "posteriors") {
      Tensor2 outputs;
      clstm.get_outputs(outputs);
      write_png((basename + ".p.png").c_str(), outputs());
    } else {
      THROW("unknown output format");
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  TRY { return main1(argc, argv); }
  CATCH(const char *message) { cerr << "FATAL: " << message << endl; }
}
