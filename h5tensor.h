// -*- C++ -*-

#ifndef h5multi_
#define h5multi_

#include <type_traits>
#include <memory>
#include <string>
#include <map>
#include <unsupported/Eigen/CXX11/Tensor>
#include "H5Cpp.h"

namespace h5tensor {
using std::string;
using std::shared_ptr;
using std::remove_reference;
using namespace H5;
template <class T, size_t n>
using Tensor = Eigen::Tensor<T, n>;
template <class T, size_t n>
using TensorRM = Eigen::Tensor<T, n, Eigen::RowMajor>;

template <class T, size_t n>
void assign(Tensor<T, n> &dest, TensorRM<T, n> &src) {
  Eigen::array<int, n> rev;
  for (int i = 0; i < n; i++) rev[i] = n - i - 1;
  dest = src.swap_layout().shuffle(rev);
}
template <class T, size_t n>
void assign(TensorRM<T, n> &dest, Tensor<T, n> &src) {
  Eigen::array<int, n> rev;
  for (int i = 0; i < n; i++) rev[i] = n - i - 1;
  dest = src.swap_layout().shuffle(rev);
}

H5::PredType pred_type(int) { return PredType::NATIVE_INT; }
H5::PredType pred_type(float) { return PredType::NATIVE_FLOAT; }
H5::PredType pred_type(double) { return PredType::NATIVE_DOUBLE; }
template <class T, int O>
void resize(Eigen::Tensor<T, 1, O> &a, hsize_t *d) {
  a.resize(int(d[0]));
}
template <class T, int O>
void resize(Eigen::Tensor<T, 2, O> &a, hsize_t *d) {
  a.resize(int(d[0]), int(d[1]));
}
template <class T, int O>
void resize(Eigen::Tensor<T, 3, O> &a, hsize_t *d) {
  a.resize(int(d[0]), int(d[1]), int(d[2]));
}
template <class T, int O>
void resize(Eigen::Tensor<T, 4, O> &a, hsize_t *d) {
  a.resize(int(d[0]), int(d[1]), int(d[2]), int(d[3]));
}

#if 0
template <class T, hsize_t n>
void ds_read(TensorRM<T, n> &a, Dataset dataset,
             initializer_list<hsize_t> offsets,
             initializer_list<hsize_t> counts) {
}
template <class T, hsize_t n>
void ds_write(const TensorRef<T, n> &a, Dataset dataset,
              initializer_list<hsize_t> offsets) {
}

void get(TensorRM<T, n> &a, const string &name) {
    DataSet dataset = h5->openDataSet(name);
    DataSpace space = dataset.getSpace();
    hsize_t offset[] = {0, 0, 0, 0, 0, 0, 0, 0};
    hsize_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};
    int rank = space.getSimpleExtentDims(count);
    if (rank != a.rank()) THROW("wrong rank");
    resize(a, (hsize_t*)count);
    space.selectHyperslab(H5S_SELECT_SET, count, offset);
    DataSpace mem(rank, count);
    mem.selectHyperslab(H5S_SELECT_SET, count, offset);
    dataset.read(a.data(), pred_type(*a.data()), mem, space);
}
#endif

struct HDF5 {
  shared_ptr<H5File> h5;
  void open(const string &name, bool rw = false, bool erase = false) {
    if (rw) {
      if (erase) {
        h5.reset(new H5File(name, H5F_ACC_TRUNC));
      } else {
        h5.reset(new H5File(name, H5F_ACC_RDWR));
      }
    } else {
      h5.reset(new H5File(name, H5F_ACC_RDONLY));
    }
  }
  ~HDF5() { h5->close(); }
  template <class T, size_t n>
  void put(TensorRM<T, n> &a, const string &name) {
    int rank = a.rank();
    DSetCreatPropList plist;  // setFillValue, etc.
    hsize_t dims[8];
    for (int i = 0; i < rank; i++) dims[i] = a.dimension(i);
    DataSpace fspace(rank, dims);
    DataSet dataset =
        h5->createDataSet(name, pred_type(*a.data()), fspace, plist);
    hsize_t start[] = {0, 0, 0, 0, 0, 0, 0, 0};
    hsize_t count[8];
    for (int i = 0; i < 8; i++) count[i] = a.dimension(i);
    DataSpace mspace(rank, dims);
    fspace.selectHyperslab(H5S_SELECT_SET, count, start);
    mspace.selectHyperslab(H5S_SELECT_SET, count, start);
    dataset.write(a.data(), pred_type(*a.data()), mspace, fspace);
  }
  template <class T, size_t n>
  void get(TensorRM<T, n> &a, const string &name) {
    DataSet dataset = h5->openDataSet(name);
    DataSpace space = dataset.getSpace();
    hsize_t offset[] = {0, 0, 0, 0, 0, 0, 0, 0};
    hsize_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};
    int rank = space.getSimpleExtentDims(count);
    if (rank != a.rank()) THROW("wrong rank");
    resize(a, (hsize_t *)count);
    space.selectHyperslab(H5S_SELECT_SET, count, offset);
    DataSpace mem(rank, count);
    mem.selectHyperslab(H5S_SELECT_SET, count, offset);
    dataset.read(a.data(), pred_type(*a.data()), mem, space);
  }
  template <class T, size_t n>
  void getrow(TensorRM<T, n> &a, int index, const string &name) {
    DataSet dataset = h5->openDataSet(name);
    DataSpace fspace = dataset.getSpace();
    hsize_t start0[] = {0, 0, 0, 0, 0, 0, 0, 0};
    hsize_t dims[] = {0, 0, 0, 0, 0, 0, 0, 0};
    int rank = fspace.getSimpleExtentDims(dims);
    if (rank != a.rank() + 1) THROW("wrong rank");
    resize(a, dims + 1);
    hsize_t count[8];
    for (int i = 0; i < 8; i++) count[i] = dims[i];
    count[0] = 1;
    hsize_t start[] = {hsize_t(index), 0, 0, 0, 0, 0, 0, 0};
    DataSpace mspace(rank, count);
    fspace.selectHyperslab(H5S_SELECT_SET, count, start);
    mspace.selectHyperslab(H5S_SELECT_SET, count, start0);
    dataset.read(a.data(), pred_type(*a.data()), mspace, fspace);
  }
  void shape(TensorRM<int, 1> &a, const string &name) {
    DataSet dataset = h5->openDataSet(name);
    DataSpace space = dataset.getSpace();
    hsize_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};
    int rank = space.getSimpleExtentDims(count);
    int n = 0;
    a.resize(rank);
    for (int i = 0; count[i]; i++) a[i] = count[i];
  }
  template <class T>
  int getvlrow(T *dest, int n, int index, const string &name) {
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
    DataType ftype(pred_type(T(0)));
    VarLenType dtype(&ftype);
    dataset.read(vl, dtype, mspace, fspace);
    T *data = (T *)vl[0].p;
    int N = vl[0].len;
    if (N > n) THROW("row too large");
    for (int i = 0; i < N; i++) dest[i] = data[i];
    dataset.vlenReclaim(dtype, mspace, DSetMemXferPropList::DEFAULT, vl);
    return N;
  }
  template <class T>
  int getvlrow1d(TensorRM<T, 1> &a, int index, const string &name) {
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
    DataType ftype(pred_type(T(0)));
    VarLenType dtype(&ftype);
    dataset.read(vl, dtype, mspace, fspace);
    T *data = (T *)vl[0].p;
    int N = vl[0].len;
    a.resize(N);
    for (int i = 0; i < N; i++) a(i) = data[i];
    dataset.vlenReclaim(dtype, mspace, DSetMemXferPropList::DEFAULT, vl);
    return N;
  }
  template <class T, size_t n>
  void getdrow(TensorRM<T, n> &a, int index, const string &name) {
    TensorRM<int, 1> dims;
    shape(dims, name);
    if (dims.size() == 1) {
#if 0
            // VLarray
            if (n == 1) {
                getvlrow1d(a, index, name);
            }
#endif
      string sname(name);
      sname += "_dims";
      TensorRM<int, 1> ndims;
      getrow(ndims, index, sname.c_str());
      if (ndims.size() != a.rank()) THROW("wrong rank (getdrow)");
      hsize_t dims[8];
      for (int i = 0; i < ndims.size(); i++) dims[i] = ndims[i];
      resize(a, dims);
      int total = 1;
      for (int i = 0; i < ndims.size(); i++) total *= ndims[i];
      // variable-shape row; take shape from _dims array
      int got = getvlrow(a.data(), total, index, name);
      if (got != total) THROW("got wrong # elements");
    } else {
      // fixed-shape row; take shape from array shape
      getrow(a, index, name);
    }
  }
  template <class T, size_t n>
  void put(Tensor<T, n> &a, const string &name) {
    TensorRM<T, n> temp;
    assign(temp, a);
    put(temp, name);
  }
  template <class T, size_t n>
  void get(Tensor<T, n> &a, const string &name) {
    TensorRM<T, n> temp;
    get(temp, name);
    assign(a, temp);
  }
  template <class T, size_t n>
  void getrow(Tensor<T, n> &a, int index, const string &name) {
    TensorRM<T, n> temp;
    getrow(temp, index, name);
    assign(a, temp);
  }
  void shape(Tensor<int, 1> &a, const string &name) {
    TensorRM<int, 1> temp;
    shape(temp, name);
    assign(a, temp);
  }
  template <class T, size_t n>
  void getdrow(Tensor<T, n> &a, int index, const string &name) {
    TensorRM<T, n> temp;
    getdrow(temp, index, name);
    assign(a, temp);
  }
};

inline HDF5 *make_HDF5() { return new HDF5(); }
}

#endif
