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
using std::set;
using std::to_string;
using std_string = std::string;
using std_wstring = std::wstring;
#define string std_string
#define wstring std_wstring

int main1(int argc, char **argv) {
  if (argc != 2) THROW("give text file as an argument");
  const char *fname = argv[1];

  string load_name = getsenv("load", "");
  if (load_name == "") THROW("must give load= parameter");
  CLSTMText clstm;
  clstm.load(load_name);

  string line;
  ifstream stream(fname);
  int output = getienv("output", 0);
  while (getline(stream, line)) {
    string orig = line + "";
    int where = line.find("\t");
    if (where >= 0) line = line.substr(0, where);
    string out = clstm.predict_utf8(line);
    if (output == 0)
      cout << out << endl;
    else if (output == 1)
      cout << line << "\t" << out << endl;
    else if (output == 2)
      cout << orig << "\t" << out << endl;
  }
  return 0;
}

int main(int argc, char **argv) {
  TRY { return main1(argc, argv); }
  CATCH(const char *message) { cerr << "FATAL: " << message << endl; }
}
