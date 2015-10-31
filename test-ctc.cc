#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <string>
#include "extras.h"
#include "clstm_compute.h"
#include <initializer_list>

using namespace ocropus;
using std::initializer_list;

void mat_of_ten2_t(Mat &result, EigenTensor2 &ten) {
  result.resize(ten.dimension(1), ten.dimension(0));
  for (int i = 0; i < result.rows(); i++)
    for (int j = 0; j < result.cols(); j++) result(i, j) = ten(j, i);
}

void ten2_of_mat_t(EigenTensor2 &result, Mat &m) {
  assert(result.dimension(0) == m.cols());
  assert(result.dimension(1) == m.rows());
  for (int i = 0; i < m.cols(); i++)
    for (int j = 0; j < m.rows(); j++) result(i, j) = m(j, i);
}

void sequence_of_tensor(Sequence &result, EigenTensor2 &data) {
  result.resize(rows(data), cols(data), 1);
  for (int i = 0; i < rows(data); i++)
    for (int j = 0; j < cols(data); j++) result[i].v(j, 0) = data(i, j);
}

void tensor_of_sequence(EigenTensor2 &result, Sequence &data) {
  result.resize(data.size(), data.rows());
  for (int i = 0; i < rows(result); i++)
    for (int j = 0; j < cols(result); j++) result(i, j) = data[i].v(j, 0);
}

void ctc_align_targets(EigenTensor2 &posteriors, EigenTensor2 &outputs,
                       EigenTensor2 &targets) {
  Sequence ps, outs, tgts;
  sequence_of_tensor(outs, outputs);
  sequence_of_tensor(tgts, targets);
  ctc_align_targets(ps, outs, tgts);
  tensor_of_sequence(posteriors, ps);
}

inline void transpose(EigenTensor2 &a) {
  Eigen::array<int, 2> axes({1, 0});
  EigenTensor2 temp = a.shuffle(axes);
  a = temp;
}

void test1() {
  EigenTensor2 outputs(3, 4);
  EigenTensor2 targets(3, 3);
  outputs.setValues({
      {1, 0, 0, 0},  //
      {0, 1, 0, 0},  //
      {0, 0, 1, 1},  //
  });
  transpose(outputs);
  targets.setValues({
      {1, 0, 0},  //
      {0, 1, 0},  //
      {0, 0, 1},  //
  });
  transpose(targets);
  EigenTensor2 result(3, 4);
  ctc_align_targets(result, outputs, targets);
  EigenTensor2 expected(3, 4);
  expected.setValues({
      {1, 0, 0, 0},  //
      {0, 1, 0, 0},  //
      {0, 0, 1, 1},  //
  });
  transpose(expected);
  Float err = reduction((expected - result).abs().maximum());
  cerr << "ctc test 1 err " << err << "\n";
  assert(err < 1e-4);
}

void test2() {
  EigenTensor2 outputs(5, 6);
  EigenTensor2 targets(5, 5);
  outputs.setValues({
      {1, .5, 0, 0, 0, 0},
      {0, .5, .5, 0, 0, 0},
      {0, 0, .5, .5, 0, 0},
      {0, 0, 0, .5, .5, 0},
      {0, 0, 0, 0, .5, 1},
  });
  transpose(outputs);
  targets.setValues({
      {1, 0, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 1, 0},
      {0, 0, 0, 0, 1},
  });
  transpose(targets);
  EigenTensor2 result(5, 6);
  ctc_align_targets(result, outputs, targets);
  EigenTensor2 expected(5, 6);
  expected.setValues({
      {1., 0.12029, 0., 0., 0., 0.},
      {0., 0.87971, 0.40013, 0., 0., 0.},
      {0., 0., 0.59987, 0.59987, 0., 0.},
      {0., 0., 0., 0.40013, 0.87971, 0.},
      {0., 0., 0., 0., 0.12029, 1.},
  });
  transpose(expected);
  Float err = reduction((expected - result).abs().maximum());
  cerr << "ctc test 2 err " << err << "\n";
  assert(err < 1e-4);
}

int main(int argc, char **argv) {
  test1();
  test2();
}
