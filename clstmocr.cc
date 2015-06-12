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
using std::set;
using std::to_string;
using std_string = std::string;
using std_wstring = std::wstring;
#define string std_string
#define wstring std_wstring

int main(int argc, char **argv) {
    if (argc != 2) THROW("give text file as an argument");
    const char *fname = argv[1];

    string load_name = getsenv("load", "");
    if (load_name == "") THROW("must give load= parameter");
    CLSTMOCR clstm;
    clstm.load(load_name);

    ifstream stream(fname);
    string line;
    while (getline(stream, line)) {
        mdarray<float> raw;
        read_png(raw, line.c_str(), true);
        for (int i = 0; i < raw.size(); i++) raw[i] = 1-raw[i];
        string out = clstm.predict_utf8(raw);
        cout << line << "\t" << out << endl;
    }
    return 0;
}
