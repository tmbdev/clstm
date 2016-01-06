#include <assert.h>
#include <math.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "clstm.h"
#include "clstm_compute.h"
#include "extras.h"
#include "utils.h"

using namespace std;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::to_string;
using std::make_pair;
using std::cout;
using std::stoi;
using namespace Eigen;
using namespace ocropus;
using std_string = std::string;
#define string std_string

typedef vector<Params> ParamVec;

double sqr(double x) { return x * x; }

double randu() {
  static double state = 0.23498023948923408293248;
  state = 179.93489901293380918 * state + 0.719408230890328424;
  state -= floor(state);
  return state;
}

double uniform(double lo = 0.0, double hi = 1.0) {
  double x = fabs(randu());
  double result = (hi - lo) * x + lo;
  PRINT(result);
  return result;
}
double exp_uniform(double lo = 1.0, double hi = 100.0) {
  assert(lo > 0 && hi > lo);
  double result = exp(uniform(log(lo), log(hi)));
  PRINT(result);
  return result;
}

void randten(Tensor2 &a, int n, int m) {
  a.resize(n, m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      a(i, j) = randu();
    }
  }
}

vector<vector<int>> conditions;

struct Timing {
  string prefix;
  Context *context;
  Tensor2 a, b, c;
  Timing() {}
  Timing(string prefix, Context *context) : prefix(prefix), context(context) {}
  void operator<<=(function<void(Tensor2 &, Tensor2 &, Tensor2 &)> f) {
    for (int i = 0; i < conditions.size(); i++) {
      double total = 0.0;
      double count = 0;
      int n = conditions[i][0];
      int l = conditions[i][1];
      int m = conditions[i][2];
      assert(n > 0 && n < 100000);
      assert(l > 0 && l < 100000);
      assert(m > 0 && m < 100000);
      for (int k = 0; k < 10; k++) {
        a.context = context;
        b.context = context;
        c.context = context;
        randten(a, n, l);
        randten(b, l, m);
        c.resize(n, m);
        double start = now();
        f(c, a, b);
        double finish = now();
        total += finish - start;
        count++;
      }
      print(prefix, n, l, m, total / count);
    }
  }
};

inline Eigen::array<Eigen::IndexPair<int>, 1> axispairs(int i, int j) {
  Eigen::array<Eigen::IndexPair<int>, 1> result = {Eigen::IndexPair<int>(i, j)};
  return result;
}

int main(int argc, char **argv) {
  int ntrial = getienv("ntrial", 1000);
  int maxmat = getienv("maxmat", 1000);
  for (int i = 0; i < ntrial; i++) {
    int n, l, m;
    n = exp_uniform(1, maxmat);
    l = exp_uniform(1, maxmat);
    m = exp_uniform(1, maxmat);
    assert(n > 0 && n < 100000);
    assert(l > 0 && l < 100000);
    assert(m > 0 && m < 100000);
    vector<int> v{n, l, m};
    conditions.push_back(v);
  }
  TRY {
    Timing nocontext("none", new Context());
    nocontext <<= [](Tensor2 &c, Tensor2 &a, Tensor2 &b) {
      c = a().contract(b(), axispairs(1, 0));
    };
    Timing threaded("threaded", new ThreadedContext(4));
    threaded <<= [](Tensor2 &c, Tensor2 &a, Tensor2 &b) {
      c = a().contract(b(), axispairs(1, 0));
    };
  }
  CATCH(const char *message) { print("ERROR", message); }
}
