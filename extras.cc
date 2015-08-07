// -*- C++ -*-

// Copyright 2006-2007 Deutsches Forschungszentrum fuer Kuenstliche Intelligenz
// or its licensors, as applicable.
// Copyright 1995-2005 Thomas M. Breuel
//
// You may not use this file except under the terms of the accompanying license.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You may
// obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

extern "C" {
#include <assert.h>
#include <math.h>
#include <unistd.h>
}

#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <map>
#include "multidim.h"
#include "pymulti.h"
#include "extras.h"

namespace ocropus {
using namespace multidim;
using namespace std;

typedef mdarray<float> floatarray;
typedef mdarray<unsigned char> bytearray;

template <class T, class S>
inline void getd0(mdarray<T> &image, mdarray<S> &slice, int index) {
  slice.resize(image.dim(1));
  for (int i = 0; i < image.dim(1); i++)
    slice.unsafe_at(i) = (S)image.unsafe_at(index, i);
}

template <class T, class S>
inline void getd1(mdarray<T> &image, mdarray<S> &slice, int index) {
  slice.resize(image.dim(0));
  for (int i = 0; i < image.dim(0); i++)
    slice.unsafe_at(i) = (S)image.unsafe_at(i, index);
}

template <class T, class S>
inline void putd0(mdarray<T> &image, mdarray<S> &slice, int index) {
  assert(slice.rank() == 1 && slice.dim(0) == image.dim(1));
  for (int i = 0; i < image.dim(1); i++)
    image.unsafe_at(index, i) = (T)slice.unsafe_at(i);
}

template <class T, class S>
inline void putd1(mdarray<T> &image, mdarray<S> &slice, int index) {
  assert(slice.rank() == 1 && slice.dim(0) == image.dim(0));
  for (int i = 0; i < image.dim(0); i++)
    image.unsafe_at(i, index) = (T)slice.unsafe_at(i);
}

/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template <class T>
void gauss1d(mdarray<T> &out, mdarray<T> &in, float sigma) {
  out.resize(in.dim(0));
  // make a normalized mask
  int range = 1 + int(3.0 * sigma);
  floatarray mask(2 * range + 1);
  for (int i = 0; i <= range; i++) {
    double y = exp(-i * i / 2.0 / sigma / sigma);
    mask(range + i) = mask(range - i) = y;
  }
  float total = 0.0;
  for (int i = 0; i < mask.dim(0); i++) total += mask(i);
  for (int i = 0; i < mask.dim(0); i++) mask(i) /= total;

  // apply it
  int n = in.size();
  for (int i = 0; i < n; i++) {
    double total = 0.0;
    for (int j = 0; j < mask.dim(0); j++) {
      int index = i + j - range;
      if (index < 0) index = 0;
      if (index >= n) index = n - 1;
      total += in(index) * mask(j);  // it's symmetric
    }
    out(i) = T(total);
  }
}

template void gauss1d(bytearray &out, bytearray &in, float sigma);
template void gauss1d(floatarray &out, floatarray &in, float sigma);

/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template <class T>
void gauss1d(mdarray<T> &v, float sigma) {
  mdarray<T> temp;
  gauss1d(temp, v, sigma);
  v.take(temp);
}

template void gauss1d(bytearray &v, float sigma);
template void gauss1d(floatarray &v, float sigma);

/// Perform 2D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template <class T>
void gauss2d(mdarray<T> &a, float sx, float sy) {
  floatarray r, s;
  for (int i = 0; i < a.dim(0); i++) {
    getd0(a, r, i);
    gauss1d(s, r, sy);
    putd0(a, s, i);
  }
  for (int j = 0; j < a.dim(1); j++) {
    getd1(a, r, j);
    gauss1d(s, r, sx);
    putd1(a, s, j);
  }
}

template void gauss2d(bytearray &image, float sx, float sy);
template void gauss2d(floatarray &image, float sx, float sy);

template <class T>
inline T &xref(mdarray<T> &a, int x, int y) {
  if (x < 0)
    x = 0;
  else if (x >= a.dim(0))
    x = a.dim(0) - 1;
  if (y < 0)
    y = 0;
  else if (y >= a.dim(1))
    y = a.dim(1) - 1;
  return a.unsafe_at(x, y);
}

template <class T>
inline T bilin(mdarray<T> &a, float x, float y) {
  int i = (int)floor(x);
  int j = (int)floor(y);
  float l = x - i;
  float m = y - j;
  float s00 = xref(a, i, j);
  float s01 = xref(a, i, j + 1);
  float s10 = xref(a, i + 1, j);
  float s11 = xref(a, i + 1, j + 1);
  return (T)((1.0 - l) * ((1.0 - m) * s00 + m * s01) +
             l * ((1.0 - m) * s10 + m * s11));
}

struct NoNormalizer : INormalizer {
  void measure(mdarray<float> &line) {}
  void normalize(mdarray<float> &out, mdarray<float> &in) {
    assert(in.dim(1) == target_height);
    out.copy(in);
  }
};

struct MeanNormalizer : INormalizer {
  double y_mean = -1;
  double y_mad = -1;
  void getparams(bool verbose) {
    vscale = getrenv("norm_vscale", 1.0);
    range = getrenv("norm_range", 1.0);
    if (verbose) print("mean_normalizer", range, vscale);
  }
  void measure(mdarray<float> &line) {
    {
      double sy = 0, s1 = 0;
      for (int i = 0; i < line.dim(0); i++) {
        for (int j = 0; j < line.dim(1); j++) {
          sy += line(i, j) * j;
          s1 += line(i, j);
        }
      }
      y_mean = sy / s1;
    }
    {
      double sy = 0, s1 = 0;
      for (int i = 0; i < line.dim(0); i++) {
        for (int j = 0; j < line.dim(1); j++) {
          sy += line(i, j) * fabs(j - y_mean);
          s1 += line(i, j);
        }
      }
      y_mad = sy / s1;
    }
  }
  void normalize(mdarray<float> &out, mdarray<float> &in) {
    float actual = vscale * 2 * range * y_mad;
    float scale = actual / target_height;
    cerr << "normalize: " << y_mean << " " << y_mad << " " << actual << endl;
    int nw = int(in.dim(0) / scale);
    int nh = target_height;
    out.resize(nw, nh);
    for (int i = 0; i < nw; i++) {
      for (int j = 0; j < nh; j++) {
        out(i, j) =
            bilin(in, scale * i, scale * (j - target_height / 2) + y_mean);
      }
    }
  }
};

void argmax1(mdarray<float> &m, mdarray<float> &a) {
  m.resize(a.dim(0));
  for (int i = 0; i < a.dim(0); i++) {
    float mv = a(i, 0);
    float mj = 0;
    for (int j = 1; j < a.dim(1); j++) {
      if (a(i, j) < mv) continue;
      mv = a(i, j);
      mj = j;
    }
    m(i) = mj;
  }
}

inline void add_smear(mdarray<float> &smooth, mdarray<float> &line) {
  int w = line.dim(0);
  int h = line.dim(1);
  for (int j = 0; j < h; j++) {
    double v = 0.0;
    for (int i = 0; i < w; i++) {
      v = v * 0.9 + line(i, j);
      smooth(i, j) += fmin(1.0, v) * 1e-3;
    }
  }
}

struct CenterNormalizer : INormalizer {
  pymulti::PyServer *py = 0;
  mdarray<float> center;
  float r = -1;
  void setPyServer(void *p) { this->py = (pymulti::PyServer *)p; }
  void getparams(bool verbose) {
    range = getrenv("norm_range", 4.0);
    smooth2d = getrenv("norm_smooth2d", 1.0);
    smooth1d = getrenv("norm_smooth1d", 0.3);
    if (verbose) print("center_normalizer", range, smooth2d, smooth1d);
  }
  void measure(mdarray<float> &line) {
    mdarray<float> smooth, smooth2;
    int w = line.dim(0);
    int h = line.dim(1);
    smooth.copy(line);
    gauss2d(smooth, h * smooth2d, h * 0.5);
    add_smear(smooth, line);  // just to avoid singularities
    mdarray<float> a(w);
    argmax1(a, smooth);
    gauss1d(center, a, h * smooth1d);
    float s1 = 0.0;
    float sy = 0.0;
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        s1 += line(i, j);
        sy += line(i, j) * fabs(j - center(i));
      }
    }
    float mad = sy / s1;
    r = int(range * mad + 1);
    if (py) {
      print("r", r);
      py->eval("ion(); clf()");
      py->eval("subplot(211)");
      py->imshowT(line, "cmap=cm.gray,interpolation='nearest'");
      py->eval("subplot(212)");
      py->imshowT(smooth, "cmap=cm.gray,interpolation='nearest'");
      py->plot(center);
      py->eval("print ginput(999)");
    }
  }
  void normalize(mdarray<float> &out, mdarray<float> &in) {
    int w = in.dim(0);
    if (w != center.dim(0)) THROW("measure doesn't match normalize");
    float scale = (2.0 * r) / target_height;
    int target_width = max(int(w / scale), 1);
    out.resize(target_width, target_height);
    for (int i = 0; i < out.dim(0); i++) {
      for (int j = 0; j < out.dim(1); j++) {
        float x = scale * i;
        float y = scale * (j - target_height / 2) + center(int(x));
        out(i, j) = bilin(in, x, y);
      }
    }
  }
};

INormalizer *make_NoNormalizer() { return new NoNormalizer(); }

INormalizer *make_MeanNormalizer() { return new MeanNormalizer(); }

INormalizer *make_CenterNormalizer() { return new CenterNormalizer(); }

INormalizer *make_Normalizer(const string &name) {
  if (name == "none") return make_NoNormalizer();
  if (name == "mean") return make_MeanNormalizer();
  if (name == "center") return make_CenterNormalizer();
  THROW("unknown normalizer name");
}

// Setting inputs/outputs using mdarray

inline void assign(Sequence &seq, mdarray<float> &a) {
  if (a.rank() == 2) {
    seq.resize(a.dim(0));
    for (int t = 0; t < seq.size(); t++) {
      seq[t].resize(a.dim(1), 1);
      for (int i = 0; i < a.dim(1); i++) seq[t](i, 0) = a(t, i);
    }
  } else if (a.rank() == 3) {
    seq.resize(a.dim(0));
    for (int t = 0; t < seq.size(); t++) {
      seq[t].resize(a.dim(1), a.dim(2));
      for (int i = 0; i < a.dim(1); i++)
        for (int j = 0; j < a.dim(2); j++) seq[t](i, j) = a(t, i, j);
    }
  } else {
    THROW("bad rank");
  }
}

void set_inputs(INetwork *net, mdarray<float> &inputs) {
  assign(net->inputs, inputs);
}
void set_targets(INetwork *net, mdarray<float> &targets) {
  assign(net->d_outputs, targets);
  for (int t = 0; t < net->outputs.size(); t++)
    net->d_outputs[t] -= net->outputs[t];
}
void set_targets_accelerated(INetwork *net, mdarray<float> &targets) {
  THROW("unimplemented");
}
void set_classes(INetwork *net, mdarray<int> &targets) {
  THROW("unimplemented");
}

// PNG I/O (taken from iulib)

#define __sigsetjmp __sigsetjump0
#include <png.h>
#undef __sigsetjmp

typedef mdarray<unsigned char> bytearray;
typedef mdarray<int> intarray;

#define ERROR(X) THROW(X)
#define CHECK_CONDITION(X)         \
  do {                             \
    if (!(X)) THROW("CHECK: " #X); \
  } while (0)
#define CHECK_ARG(X)                   \
  do {                                 \
    if (!(X)) THROW("CHECK_ARG: " #X); \
  } while (0)

bool png_flip = false;

void read_png(bytearray &image, FILE *fp, bool gray) {
  int d;
  int spp;
  int png_transforms;
  int num_palette;
  png_byte bit_depth, color_type, channels;
  int w, h, rowbytes;
  png_bytep rowptr;
  png_bytep *row_pointers;
  png_structp png_ptr;
  png_infop info_ptr, end_info;
  png_colorp palette;

  if (!fp) ERROR("fp not defined");

  // Allocate the 3 data structures
  if ((png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
                                        NULL, NULL)) == NULL)
    ERROR("png_ptr not made");

  if ((info_ptr = png_create_info_struct(png_ptr)) == NULL) {
    png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    ERROR("info_ptr not made");
  }

  if ((end_info = png_create_info_struct(png_ptr)) == NULL) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    ERROR("end_info not made");
  }

  // Set up png setjmp error handling

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    ERROR("internal png error");
  }

  png_init_io(png_ptr, fp);

  // Set the transforms flags. Whatever you do here,
  // DO NOT invert binary using PNG_TRANSFORM_INVERT_MONO!!
  // To remove alpha channel, use PNG_TRANSFORM_STRIP_ALPHA
  // To strip 16 --> 8 bit depth, use PNG_TRANSFORM_STRIP_16 */
  //#if 0 /* this does both */
  //        png_transforms = PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_STRIP_ALPHA;
  //#else /* this just strips alpha */
  //        png_transforms = PNG_TRANSFORM_STRIP_ALPHA;
  //#endif
  png_transforms = PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_STRIP_ALPHA |
                   PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND;

  // Do it!
  png_read_png(png_ptr, info_ptr, png_transforms, NULL);

  row_pointers = png_get_rows(png_ptr, info_ptr);
  w = png_get_image_width(png_ptr, info_ptr);
  h = png_get_image_height(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  channels = png_get_channels(png_ptr, info_ptr);

  spp = channels;

  if (spp == 1) {
    d = bit_depth;
  } else if (spp == 2) {
    d = 2 * bit_depth;
    ERROR("there shouldn't be 2 spp!");
  } else if (spp == 3) {
    d = 4 * bit_depth;
  } else { /* spp == 4 */
    d = 4 * bit_depth;
    ERROR("there shouldn't be 4 spp!");
  }

  /* Remove if/when this is implemented for all bit_depths */
  if (spp == 3 && bit_depth != 8) {
    fprintf(stderr, "Help: spp = 3 and depth = %d != 8\n!!", bit_depth);
    ERROR("not implemented for this depth");
  }

  intarray color_map;

  if (color_type == PNG_COLOR_TYPE_PALETTE ||
      color_type == PNG_COLOR_MASK_PALETTE) { /* generate a colormap */
    png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette);
    color_map.resize(3, num_palette);
    for (int cindex = 0; cindex < num_palette; cindex++) {
      color_map(0, cindex) = palette[cindex].red;
      color_map(1, cindex) = palette[cindex].green;
      color_map(2, cindex) = palette[cindex].blue;
    }
  }

  if (gray)
    image.resize(w, h);
  else
    image.resize(w, h, 3);

  if (spp == 1) {
    CHECK_CONDITION(color_type != PNG_COLOR_TYPE_PALETTE &&
                    color_type != PNG_COLOR_MASK_PALETTE);
    CHECK_CONDITION(bit_depth == 1 || bit_depth == 8);
    for (int i = 0; i < h; i++) {
      rowptr = row_pointers[i];
      for (int j = 0; j < w; j++) {
        int x = j;
        int y = png_flip ? (h - i - 1) : i;
        int value;
        if (bit_depth == 1) {
          value = (rowptr[j / 8] & (128 >> (j % 8))) ? 255 : 0;
        } else {
          value = rowptr[j];
        }
        if (gray) {
          image(x, y) = value;
        } else {
          image(x, y, 0) = value;
          image(x, y, 1) = value;
          image(x, y, 2) = value;
        }
      }
    }
  } else {
    CHECK_CONDITION(color_type != PNG_COLOR_TYPE_PALETTE &&
                    color_type != PNG_COLOR_MASK_PALETTE);
    CHECK_CONDITION(bit_depth == 8);
    for (int i = 0; i < h; i++) {
      rowptr = row_pointers[i];
      int k = 0;
      for (int j = 0; j < w; j++) {
        int x = j;
        int y = png_flip ? (h - i - 1) : i;
        if (gray) {
          int value = rowptr[k++];
          value += rowptr[k++];
          value += rowptr[k++];
          image(x, y) = value / 3;
        } else {
          image(x, y, 0) = rowptr[k++];
          image(x, y, 1) = rowptr[k++];
          image(x, y, 2) = rowptr[k++];
        }
      }
    }
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
}

void write_png(FILE *fp, bytearray &image) {
  // int d;
  png_byte bit_depth, color_type;
  int w, h;
  png_structp png_ptr;
  png_infop info_ptr;
  unsigned int default_xres = 300;
  unsigned int default_yres = 300;

  int rank = image.rank();
  CHECK_ARG(image.rank() == 2 || (image.rank() == 3 && image.dim(2) == 3));

  if (!fp) ERROR("stream not open");

  /* Allocate the 2 data structures */
  if ((png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
                                         NULL, NULL)) == NULL)
    ERROR("png_ptr not made");

  if ((info_ptr = png_create_info_struct(png_ptr)) == NULL) {
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    ERROR("info_ptr not made");
  }

  /* Set up png setjmp error handling */
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    ERROR("internal png error");
  }

  png_init_io(png_ptr, fp);

  w = image.dim(0);
  h = image.dim(1);
  // d = image.dim(2);
  bit_depth = 8;
  color_type = PNG_COLOR_TYPE_RGB;

  png_set_IHDR(png_ptr, info_ptr, w, h, bit_depth, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);
  png_set_pHYs(png_ptr, info_ptr, default_xres, default_yres,
               PNG_RESOLUTION_METER);
  png_write_info(png_ptr, info_ptr);

  bytearray rowbuffer;
  rowbuffer.resize(3 * w);
  for (int i = 0; i < h; i++) {
    int k = 0;
    for (int j = 0; j < w; j++) {
      int x = j;
      int y = png_flip ? (h - i - 1) : i;
      if (rank == 2) {
        int value = image(x, y);
        rowbuffer(k++) = value;
        rowbuffer(k++) = value;
        rowbuffer(k++) = value;
      } else {
        rowbuffer(k++) = image(x, y, 0);
        rowbuffer(k++) = image(x, y, 1);
        rowbuffer(k++) = image(x, y, 2);
      }
    }

    png_byte *p = &rowbuffer(0);
    png_write_rows(png_ptr, &p, 1);
  }

  png_write_end(png_ptr, info_ptr);

  png_destroy_write_struct(&png_ptr, &info_ptr);
}

void read_png(bytearray &image, const char *name, bool gray) {
  FILE *stream = fopen(name, "r");
  if (!stream) THROW("error on open");
  read_png(image, stream, gray);
  fclose(stream);
}
void write_png(const char *name, bytearray &image) {
  FILE *stream = fopen(name, "w");
  if (!stream) THROW("error on open");
  write_png(stream, image);
  fclose(stream);
}

template <class U, class T>
inline void copy_scale(mdarray<U> &it, mdarray<T> &other, double scale) {
  it.clear();
  it.allocate(other.total);
  for (int i = 0; i < mdarray<T>::MAXRANK + 1; i++) it.dims[i] = other.dims[i];
  it.fill = other.fill;
  for (int i = 0; i < it.fill; i++) it.data[i] = other.data[i] * scale;
}

void read_png(mdarray<float> &image, FILE *fp, bool gray) {
  mdarray<unsigned char> temp;
  read_png(temp, fp, gray);
  copy_scale(image, temp, 1.0 / 255.0);
}

void write_png(FILE *fp, mdarray<float> &image) {
  mdarray<unsigned char> temp;
  copy_scale(temp, image, 255.0);
  write_png(fp, temp);
}

void read_png(mdarray<float> &image, const char *name, bool gray) {
  mdarray<unsigned char> temp;
  read_png(temp, name, gray);
  copy_scale(image, temp, 1.0 / 255.0);
}

void write_png(const char *name, mdarray<float> &image) {
  mdarray<unsigned char> temp;
  copy_scale(temp, image, 255.0);
  write_png(name, temp);
}

#if 0
Network make_net_init(const string &kind, int nclasses, int dim, string prefix) {
    int nhidden = getrenv((prefix+"hidden").c_str(), 100);
    if (kind == "bidi2") {
        int nhidden2 = getrenv((prefix+"hidden2").c_str(), -1);
        net->init(nclasses, nhidden2, nhidden, dim);
        print("init-bidi2", nclasses, nhidden2, nhidden, dim);
    } else {
        net->init(nclasses, nhidden, dim);
        print("init", nclasses, nhidden, dim);
    }
    return net;
}
#endif

void glob(vector<string> &result, const string &arg) {
  result.clear();
  glob_t buf;
  glob(arg.c_str(), GLOB_TILDE, nullptr, &buf);
  for (int i = 0; i < buf.gl_pathc; i++) {
    result.push_back(buf.gl_pathv[i]);
  }
  if (buf.gl_pathc > 0) globfree(&buf);
}

unsigned long random_state;

void srandomize() {
  random_state = getienv("seed", 0);
  if (random_state == 0) {
    random_state = (unsigned long)fmod(now() * 1e6, 1e9);
    char **ep = environ;
    while (*ep) {
      char *p = *ep++;
      while (*p) random_state = 17 * random_state + *p++;
    }
  }
  // cerr << "# srandomize " << random_state << endl;
}

void randstep() {
  random_state = (random_state * 1664525 + 1013904223) % (1ul << 32);
}

unsigned urandom() {
  randstep();
  return random_state;
}

int irandom() {
  randstep();
  return abs(int(random_state));
}
double drandom() { return double(irandom() % 999999733) / 999999733.0; }
}
