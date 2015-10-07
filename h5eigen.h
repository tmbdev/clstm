// -*- C++ -*-

#ifndef h5eigen_
#define h5eigen_

#include <type_traits>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <map>
#include "H5Cpp.h"

namespace h5eigen {
using std::string;
using std::shared_ptr;
using std::remove_reference;
using std::map;
using namespace H5;
using namespace Eigen;

struct HDF5 {
  shared_ptr<H5File> h5;
  void open(const char *name, bool rw = false) {
    if (rw) {
      h5.reset(new H5File(name, H5F_ACC_TRUNC));
    } else {
      h5.reset(new H5File(name, H5F_ACC_RDONLY));
    }
  }
  ~HDF5() { h5->close(); }
  H5::PredType pred_type(int) { return PredType::NATIVE_INT; }
  H5::PredType pred_type(float) { return PredType::NATIVE_FLOAT; }
  H5::PredType pred_type(double) { return PredType::NATIVE_DOUBLE; }
  bool exists(const char *name) {
    try {
      DataSet dataset = h5->openDataSet(name);
      return true;
    } catch (FileIException e) {
      return false;
    } catch (GroupIException e) {
      return false;
    }
  }
  template <class T>
  void get(T &a, const char *name) {
    DataSet dataset = h5->openDataSet(name);
    DataSpace space = dataset.getSpace();
    hsize_t offset[] = {0, 0, 0, 0, 0, 0};
    hsize_t count[] = {0, 0, 0, 0, 0, 0};
    int rank = space.getSimpleExtentDims(count);
    Matrix<float, Dynamic, Dynamic, RowMajor> temp;
    if (rank == 1)
      temp.resize(count[0], 1);
    else if (rank == 2)
      temp.resize(count[0], count[1]);
    else
      THROW("unsupported rank");
    space.selectHyperslab(H5S_SELECT_SET, count, offset);
    DataSpace mem(rank, count);
    mem.selectHyperslab(H5S_SELECT_SET, count, offset);
    dataset.read(&temp(0, 0), pred_type(temp(0, 0)), mem, space);
    a = temp;
  }
  template <class T>
  void get1d(T &a, const char *name) {
    DataSet dataset = h5->openDataSet(name);
    DataSpace space = dataset.getSpace();
    hsize_t offset[] = {0, 0, 0, 0, 0, 0};
    hsize_t count[] = {0, 0, 0, 0, 0, 0};
    int rank = space.getSimpleExtentDims(count);
    if (rank == 1)
      a.resize(count[0]);
    else if (rank == 2)
      a.resize(count[0] * count[1]);
    else if (rank == 3)
      a.resize(count[0] * count[1] * count[2]);
    else if (rank == 4)
      a.resize(count[0] * count[1] * count[2] * count[3]);
    else
      THROW("unsupported rank");
    space.selectHyperslab(H5S_SELECT_SET, count, offset);
    DataSpace mem(rank, count);
    mem.selectHyperslab(H5S_SELECT_SET, count, offset);
    dataset.read(&a[0], pred_type(a[0]), mem, space);
  }
  template <class T>
  void put(T &a, const char *name, int rank = 2) {
    Matrix<float, Dynamic, Dynamic, RowMajor> temp = a;
    DSetCreatPropList plist;  // setFillValue, etc.
    hsize_t rows = ROWS(temp);
    hsize_t cols = COLS(temp);
    hsize_t dim[] = {rows, cols, 0, 0, 0, 0};
    DataSpace fspace(rank, dim);
    DataSet dataset =
        h5->createDataSet(name, pred_type(temp(0, 0)), fspace, plist);
    hsize_t start[] = {0, 0, 0, 0, 0, 0};
    hsize_t count[] = {rows, cols, 0, 0, 0, 0};
    DataSpace mspace(rank, dim);
    fspace.selectHyperslab(H5S_SELECT_SET, count, start);
    mspace.selectHyperslab(H5S_SELECT_SET, count, start);
    dataset.write(&temp(0, 0), pred_type(temp(0, 0)), mspace, fspace);
  }
  template <class T>
  void getvlrow(T &a, int index, const char *name) {
    typedef typename remove_reference<decltype(a[0])>::type S;
    DataSet dataset = h5->openDataSet(name);
    DataSpace space = dataset.getSpace();
    hsize_t dims[] = {0, 0, 0, 0};
    int rank = space.getSimpleExtentDims(dims);
    if (rank != 1) THROW("wrong rank");
    hsize_t start0[] = {0, 0};
    hsize_t start[] = {hsize_t(index), 0};
    hsize_t count[] = {1, 0};
    DataSpace fspace(1, dims);
    DataSpace mspace(1, count);
    fspace.selectHyperslab(H5S_SELECT_SET, count, start);
    mspace.selectHyperslab(H5S_SELECT_SET, count, start0);
    hvl_t vl[1];
    DataType ftype(pred_type(S(0)));
    VarLenType dtype(&ftype);
    dataset.read(vl, dtype, mspace, fspace);
    S *data = (S *)vl[0].p;
    int N = vl[0].len;
    a.resize(N, 1);
    for (int i = 0; i < N; i++) a(i, 0) = data[i];
    dataset.vlenReclaim(dtype, mspace, DSetMemXferPropList::DEFAULT, vl);
  }
  void attr.set(string name, string value) {
    Group root(h5->openGroup("/"));
    StrType strdatatype(PredType::C_S1, 256);
    DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
    H5std_string buffer(value);
    root.createAttribute(name, strdatatype, attr_dataspace)
        .write(strdatatype, buffer);
  }
  string attr.get(string name) {
    Group root(h5->openGroup("/"));
    StrType strdatatype(PredType::C_S1, 256);
    DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
    H5std_string buffer;
    root.createAttribute(name, strdatatype, attr_dataspace)
        .read(strdatatype, buffer);
    return buffer;
  }
  void attr.gets(map<string, string> &result) {
    Group root(h5->openGroup("/"));
    StrType strdatatype(PredType::C_S1, 256);
    DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
    H5std_string buffer;
    for (int i = 0; i < root.getNumAttrs(); i++) {
      Attribute a = root.openAttribute(i);
      string key = a.getName();
      a.read(strdatatype, buffer);
      result[key] = buffer;
    }
  }
};

inline HDF5 *make_HDF5() { return new HDF5(); }
}

#endif
