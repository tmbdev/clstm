// -*- C++ -*-

// A simple and sane multidimensional array class.

#ifndef multidim_
#define multidim_

#define MDSTR(X) #X
#define MDSTR1(X) MDSTR(X)
#define MDCHECK(X)                                          \
  while (!(X)) {                                            \
    THROW("FAILED: " __FILE__ ":" MDSTR1(__LINE__) ":" #X); \
  }

#include <stdlib.h>

namespace multidim {
template <class T>
struct mdarray {
  static const int MAXRANK = 8;
  int dims[MAXRANK + 1] = {0, 0};
  int total = 0;
  int fill = 0;
  T *data = 0;
  bool owned = false;

  void take(mdarray<T> &other) {
    for (int i = 0; i < MAXRANK + 1; i++) dims[i] = other.dims[i];
    total = other.total;
    fill = other.fill;
    data = other.data;
    owned = other.owned;
    for (int i = 0; i < MAXRANK + 1; i++) other.dims[i] = 0;
    other.total = 0;
    other.fill = 0;
    other.data = 0;
    other.owned = false;
  }

  // cleared array by default
  mdarray() {}

  // deallocate data on deletion
  ~mdarray() { clear(); }

  // create an array of the given size
  template <typename... Args>
  mdarray(Args... args) {
    resize(args...);
  }

  // alias an array to a given pointer
  template <typename... Args>
  mdarray(T *p, int *dims) {
    alias_(p, dims);
  }

  // no copy constructors or assignment
  mdarray(mdarray<T> &) = delete;
  mdarray(const mdarray<T> &) = delete;

  // clear all allocated data
  void clear() {
    if (owned && data) delete[] data;
    data = 0;
    total = 0;
    fill = 0;
    for (int i = 0; i < MAXRANK; i++) dims[i] = 0;
  }

  // allocate n elements
  void allocate(int n) {
    MDCHECK(!data);
    data = new T[n];
    total = n;
    fill = 0;
    owned = true;
  }

  // copy another array
  void copy(mdarray<T> &other) {
    clear();
    allocate(other.total);
    for (int i = 0; i < MAXRANK + 1; i++) dims[i] = other.dims[i];
    fill = other.fill;
    for (int i = 0; i < fill; i++) data[i] = other.data[i];
  }

  // get the extent of dimension i
  int dim(int i) {
    MDCHECK(unsigned(i) < MAXRANK);
    MDCHECK(dims[i] > 0);
    return dims[i];
  }

  // get the rank of the array
  int rank() {
    for (int i = 0; i < MAXRANK + 1; i++)
      if (!dims[i]) return i;
    MDCHECK(MAXRANK < MAXRANK);
  }

  // total number of elements in linearized array
  int size() { return fill; }

  // multidimensional subscripting
  template <typename... Args>
  T &operator()(Args... args) {
    int indexes[] = {args...};
    int rank = sizeof indexes / sizeof indexes[0];
    MDCHECK(rank > 0 && rank <= MAXRANK);
    MDCHECK(dims[rank] == 0);
    MDCHECK(unsigned(indexes[0]) < unsigned(dims[0]));
    int index = indexes[0];
    for (int i = 1; i < rank; i++) {
      MDCHECK(unsigned(indexes[i]) < unsigned(dims[i]));
      index = index * dims[i] + indexes[i];
    }
    return data[index];
  }
  // multidimensional subscripting
  template <typename... Args>
  T &unsafe_at(Args... args) {
    int indexes[] = {args...};
    int rank = sizeof indexes / sizeof indexes[0];
    int index = indexes[0];
    for (int i = 1; i < rank; i++) {
      index = index * dims[i] + indexes[i];
    }
    return data[index];
  }
  T &operator[](int i) {
    MDCHECK(unsigned(i) < unsigned(total));
    return data[i];
  }
  // multidimensional subscripting
  template <typename... Args>
  void getSlice(mdarray<T> &result, Args... args) {
    int indexes[] = {args...};
    int rank = sizeof indexes / sizeof indexes[0];
    MDCHECK(rank > 0 && rank <= MAXRANK);
    MDCHECK(dims[rank] != 0);
    MDCHECK(unsigned(indexes[0]) < unsigned(dims[0]));
    int index = indexes[0];
    for (int i = 1; i < rank; i++) {
      MDCHECK(unsigned(indexes[i]) < unsigned(dims[i]));
      index = index * dims[i] + indexes[i];
    }
    result.alias_(data + index, dims + rank);
  }

  // reshape the array without changing its contents
  template <typename... Args>
  mdarray &reshape(Args... args) {
    int indexes[MAXRANK + 1] = {args...};
    int rank = sizeof indexes / sizeof indexes[0];
    indexes[rank] = 0;
    reshape_(indexes);
    return *this;
  }

  // alias an array to given data; the pointer will not
  // get deleted on destruction
  template <typename... Args>
  mdarray &alias(T *p, Args... args) {
    int indexes[MAXRANK + 1] = {args...};
    int rank = sizeof indexes / sizeof indexes[0];
    indexes[rank] = 0;
    alias_(p, indexes);
    return *this;
  }

  // resize the array, destroying the data it contains
  template <typename... Args>
  mdarray &resize(Args... args) {
    int indexes_[] = {args...};
    int rank = sizeof indexes_ / sizeof indexes_[0];
    int indexes[MAXRANK + 1] = {args...};
    indexes[rank] = 0;
    resize_(indexes);
    return *this;
  }

  // fill the array with the given value
  template <class S>
  mdarray &constant(S value) {
    for (int i = 0; i < total; i++) data[i] = value;
    return *this;
  }

  // equivalent shape-related functions using a pointer
  // to an integral, zero-terminated array

  template <typename INT>
  void reshape_(INT *shape, bool exact = true) {
    int nsize = unsigned(prod_(shape));
    MDCHECK(nsize <= total);
    if (exact) MDCHECK(nsize == prod_(dims));
    int i;
    for (i = 0; i < MAXRANK; i++) {
      if (!shape[i]) break;
      dims[i] = int(shape[i]);
    }
    MDCHECK(i <= MAXRANK);
    for (; i <= MAXRANK; i++) dims[i] = 0;
    fill = prod_(dims);
  }
  template <typename INT>
  void resize_(INT *shape) {
    int nsize = prod_(shape);
    if (nsize < total) {
      // nothing
    } else {
      clear();
      allocate(nsize);
    }
    reshape_(shape, false);
  }
  template <typename INT>
  void alias_(T *p, INT *shape) {
    data = p;
    owned = false;
    total = prod_(shape);
    reshape_(shape);
  }
  template <typename INT>
  static inline int prod_(INT *p) {
    if (p[0] == 0) return 0;
    int total = 1;
    for (int i = 0; p[i]; i++) total *= p[i];
    return total;
  }

  // internal consistency check, invariants
  void check_() {
    MDCHECK(unsigned(rank()) <= MAXRANK);
    MDCHECK(prod_(dims) == total);
  }

  // assignment and arithmetic
  template <class O>
  void operator=(mdarray<O> &other) {
    copy(other);
  }
  template <class O>
  void operator=(const mdarray<O> &other) {
    copy(other);
  }

  template <class O, class P>
  void clip(const O &lo, const P &hi) {
    for (int i = 0; i < fill; i++) {
      T value = data[i];
      if (value < lo)
        data[i] = lo;
      else if (value > hi)
        data[i] = hi;
    }
  }
  T min() {
    T result = data[0];
    for (int i = 1; i < fill; i++) {
      T value = data[i];
      if (value < result) result = value;
    }
    return result;
  }
  T max() {
    T result = data[0];
    for (int i = 1; i < fill; i++) {
      T value = data[i];
      if (value > result) result = value;
    }
    return result;
  }
  double normsq() {
    double result = 0.0;
    for (int i = 0; i < fill; i++) {
      T value = data[i];
      result += value * value;
    }
    return result;
  }
  double norm() { return sqrt(normsq()); }
  void randomize() {
    for (int i = 0; i < fill; i++) data[i] = drand48();
  }

  template <class O>
  void operator+=(const O &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] += other.data[i];
  }
  template <class O>
  void operator-=(const O &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] += other.data[i];
  }
  template <class O>
  void operator*=(const O &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] *= other.data[i];
  }
  template <class O>
  void operator/=(const O &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] /= other.data[i];
  }

  template <class O>
  void operator+=(const mdarray<O> &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] += other.data[i];
  }
  template <class O>
  void operator-=(const mdarray<O> &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] += other.data[i];
  }
  template <class O>
  void operator*=(const mdarray<O> &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] *= other.data[i];
  }
  template <class O>
  void operator/=(const mdarray<O> &other) {
    assert(fill == other.fill);
    for (int i = 0; i < fill; i++) data[i] /= other.data[i];
  }
};

template <class T, class S>
inline void to_eigen_matrix(T result, mdarray<S> &a) {
  MDCHECK(a.rank() == 2);
  result.resize(a.dim(0), a.dim(1));
  for (int i = 0; i < a.dim(0); i++)
    for (int j = 0; j < a.dim(1); j++) result(i, j) = a(i, j);
}

template <class T, class S>
inline void to_eigen_vector(T result, mdarray<S> &a) {
  MDCHECK(a.rank() == 1);
  result.resize(a.dim(0));
  for (int i = 0; i < a.dim(0); i++) result(i) = a(i);
}

template <class T, class S>
inline void from_eigen_matrix(mdarray<S> &result, T a) {
  result.resize(ROWS(a), COLS(a));
  for (int i = 0; i < ROWS(a); i++)
    for (int j = 0; j < COLS(a); j++) result(i, j) = a(i, j);
}
}

#endif
