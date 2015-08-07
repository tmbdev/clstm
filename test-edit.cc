#include "clstm.h"
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char **argv) {
  ifstream file1(argv[1]);
  ifstream file2(argv[2]);

  for (;;) {
    string line1, line2;
    if (file1.eof() || file2.eof()) break;
    getline(file1, line1);
    getline(file2, line2);
    int err = levenshtein(line1, line2);
    cout << err << "\t";
    cout << line1 << "\t";
    cout << line2 << "\n";
  }
}
