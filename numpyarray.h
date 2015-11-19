#ifndef clstm_numpyarray_
#define clstm_numpyarray_

#include "numpy/arrayobject.h"

template <class T, int TYPENUM>
struct NumPyArray {
  PyArrayObject *obj = 0;
  NumPyArray() {}
  NumPyArray(PyObject *object_) {
    if (!object_) throw "null pointer";
    if (!PyArray_Check(object_)) throw "expected a numpy array";
    obj = (PyArrayObject *)object_;
    Py_INCREF(obj);
    valid();
  }
  NumPyArray(NumPyArray<T, TYPENUM> &other) {
    Py_INCREF(other.obj);
    Py_DECREF(obj);
    obj = other.obj;
  }
  NumPyArray(int d0, int d1 = 0, int d2 = 0, int d3 = 0) {
    npy_intp ndims[] = {d0, d1, d2, d3, 0};
    int rank = 0;
    while (ndims[rank]) rank++;
    obj = PyArray_SimpleNew(rank, ndims, TYPENUM);
    valid();
  }
  ~NumPyArray() {
    Py_DECREF(obj);
    obj = 0;
  }
  void operator=(NumPyArray<T, TYPENUM> &other) {
    Py_INCREF(other.obj);
    Py_DECREF(obj);
    obj = other.obj;
  }
  void valid() {
    if (!obj) throw "no array set";
    if (PyArray_TYPE(obj) != TYPENUM) throw "wrong numpy array type";
    if ((PyArray_FLAGS(obj) & NPY_ARRAY_C_CONTIGUOUS) == 0)
      throw "expected contiguous array";
  }
  int rank() {
    valid();
    return PyArray_NDIM(obj);
  }
  int dim(int i) {
    valid();
    return PyArray_DIM(obj, i);
  }
  int size() {
    valid();
    return PyArray_SIZE(obj);
  }
  void resize(int d0, int d1 = 0, int d2 = 0, int d3 = 0) {
    npy_intp ndims[] = {d0, d1, d2, d3, 0};
    int rank = 0;
    while (ndims[rank]) rank++;
    PyArray_Dims dims = {ndims, rank};
    if (PyArray_Resize(obj, &dims, 0, NPY_CORDER) == nullptr)
      throw "resize failed";
  }
  T &operator()(int i) {
    assert(rank() == 1);
    assert(unsigned(i) < unsigned(dim(0)));
    T *data = (T *)PyArray_DATA(obj);
    return data[i];
  }
  T &operator()(int i, int j) {
    assert(rank() == 2);
    assert(unsigned(i) < unsigned(dim(0)));
    assert(unsigned(j) < unsigned(dim(1)));
    T *data = (T *)PyArray_DATA(obj);
    return data[i * dim(1) + j];
  }
  T &operator()(int i, int j, int k) {
    assert(rank() == 3);
    assert(unsigned(i) < unsigned(dim(0)));
    assert(unsigned(j) < unsigned(dim(1)));
    assert(unsigned(k) < unsigned(dim(2)));
    T *data = (T *)PyArray_DATA(obj);
    return data[(i * dim(1) + j) * dim(2) + k];
  }
  T &operator()(int i, int j, int k, int l) {
    assert(rank() == 4);
    assert(unsigned(i) < unsigned(dim(0)));
    assert(unsigned(j) < unsigned(dim(1)));
    assert(unsigned(k) < unsigned(dim(2)));
    assert(unsigned(l) < unsigned(dim(3)));
    T *data = (T *)PyArray_DATA(obj);
    return data[((i * dim(1) + j) * dim(2) + k) * dim(3) + l];
  }
  T *data() {
    valid();
    return (T *)PyArray_DATA(obj);
  }
  void copyTo(T *dest) {
    valid();
    T *data = (T *)PyArray_DATA(obj);
    int N = size();
    for (int i = 0; i < N; i++) dest[i] = data[i];
  }
};

typedef NumPyArray<float, NPY_FLOAT> npa_float;
#endif
