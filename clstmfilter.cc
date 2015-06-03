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
    if (argc != 2) throw "give text file as an argument";
    const char *fname = argv[1];

    string load_name = getsenv("load", "");
    if (load_name == "") throw "must give load= parameter";
    CLSTMText clstm;
    clstm.load(load_name);

    string line;
    ifstream stream(fname);
    int trial = 0;
    int output = getienv("output", 0);
    while (getline(stream, line)) {
        string orig = line+"";
        int where = line.find("\t");
        if (where >= 0) line = line.substr(0, where);
        string out = clstm.predict_utf8(line);
        if (output == 0) cout << out << endl;
        else if (output == 1) cout << line << "\t" << out << endl;
        else if (output == 2) cout << orig << "\t" << out << endl;
    }
    return 0;
}
