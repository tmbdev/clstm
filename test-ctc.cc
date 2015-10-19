#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include "extras.h"

using namespace ocropus;

void mat_of_ten2_t(Mat &result, Tensor2 &ten) {
  result.resize(ten.dimension(1), ten.dimension(0));
  for (int i = 0; i < result.rows(); i++)
    for (int j = 0; j < result.cols(); j++) result(i, j) = ten(j, i);
}

void ten2_of_mat_t(Tensor2 &result, Mat &m) {
  assert(result.dimension(0)==m.cols());
  assert(result.dimension(1)==m.rows());
  for (int i = 0; i < m.cols(); i++)
    for (int j = 0; j < m.rows(); j++) result(i, j) = m(j, i);
}

void ctc_align_targets(Tensor2 &posteriors, Tensor2 &outputs, Tensor2 &targets) {
  Mat ps, outs, tgts;
  mat_of_ten2_t(outs, outputs);
  mat_of_ten2_t(tgts, targets);
  ctc_align_targets(ps, outs, tgts);
  ten2_of_mat_t(posteriors, ps);
}

void test1() {
  Tensor2 outputs(3, 4);
  Tensor2 targets(3, 3);
  outputs.setValues({
      {1, 0, 0, 0},  //
      {0, 1, 0, 0},  //
      {0, 0, 1, 1},  //
  });
  targets.setValues({
      {1, 0, 0},  //
      {0, 1, 0},  //
      {0, 0, 1},  //
  });
  Tensor2 result(3,4);
  ctc_align_targets(result, outputs, targets);
  Tensor2 expected(3, 4);
  expected.setValues({
      {1, 0, 0, 0},  //
      {0, 1, 0, 0},  //
      {0, 0, 1, 1},  //
  });
  Tensor1 err = (expected-result).abs().maximum();
  cerr << "test1 " << err(0) << "\n";
  assert(err(0) < 1e-4);
}

void test2() {
  Tensor2 outputs(5, 6);
  Tensor2 targets(5, 5);
  outputs.setValues({
      {1, .5, 0, 0, 0, 0},
      {0, .5, .5, 0, 0, 0},
      {0, 0, .5, .5, 0, 0},
      {0, 0, 0, .5, .5, 0},
      {0, 0, 0, 0, .5, 1},
  });
  targets.setValues({
      {1, 0, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 1, 0},
      {0, 0, 0, 0, 1},
  });
  Tensor2 result(5, 6);
  ctc_align_targets(result, outputs, targets);
  Tensor2 expected(5, 6);
  expected.setValues({
      {1., 0.12029, 0., 0., 0., 0.},
      {0., 0.87971, 0.40013, 0., 0., 0.},
      {0., 0., 0.59987, 0.59987, 0., 0.},
      {0., 0., 0., 0.40013, 0.87971, 0.},
      {0., 0., 0., 0., 0.12029, 1.},
  });
  Tensor1 err = (expected-result).abs().maximum();
  cerr << "test2 " << err(0) << "\n";
  assert(err(0) < 1e-4);
}

int main(int argc, char **argv) {
  test1();
  test2();
}
