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

// FIXME: speed up convolutions, maybe using Eigen built-in

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

extern "C" {
#include <assert.h>
#include <math.h>
#include <unistd.h>
#define __sigsetjmp __sigsetjump0
#ifdef GOOGLE
#include "third_party/png/png.h"
#else
#include <png.h>
#endif
#undef __sigsetjmp
}

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "extras.h"
#include "pytensor.h"
#include "utils.h"

namespace ocropus {
using namespace std;

template <class T, int N>
inline TensorMap<Tensor<T, N>> TM(Tensor<T, N> &t) {
  return TensorMap<Tensor<T, N>>(t.data(), t.dimensions());
}

/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template <class T>
void gauss1d(Tensor<T, 1> &out, Tensor<T, 1> &in, float sigma) {
  out.resize(in.dimension(0));
  // make a normalized mask
  int range = 1 + int(3.0 * sigma);
  Tensor<float, 1> mask(2 * range + 1);
  for (int i = 0; i <= range; i++) {
    double y = exp(-i * i / 2.0 / sigma / sigma);
    mask(range + i) = mask(range - i) = y;
  }
  float total = 0.0;
  for (int i = 0; i < mask.dimension(0); i++) total += mask(i);
  for (int i = 0; i < mask.dimension(0); i++) mask(i) /= total;

  T *in_ = in.data();
  float *mask_ = mask.data();
  // apply it
  int n = in.size();
  int m = mask.dimension(0);
  for (int i = 0; i < n; i++) {
    double total = 0.0;
    for (int j = 0; j < m; j++) {
      int index = i + j - range;
      if (index < 0) index = 0;
      if (index >= n) index = n - 1;
      total += in_[index] * mask_[j];  // it's symmetric
    }
    out(i) = T(total);
  }
}

template void gauss1d(Tensor<unsigned char, 1> &out,
                      Tensor<unsigned char, 1> &in, float sigma);
template void gauss1d(Tensor<float, 1> &out, Tensor<float, 1> &in, float sigma);

/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template <class T>
void gauss1d(Tensor<T, 1> &v, float sigma) {
  Tensor<T, 1> temp;
  gauss1d(temp, v, sigma);
  v = temp;
}

template void gauss1d(Tensor<unsigned char, 1> &v, float sigma);
template void gauss1d(Tensor<float, 1> &v, float sigma);

/// Perform 2D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

void gauss2d(TensorMap2 a, float sx, float sy) {
  Tensor<Float, 1> r, s;
  for (int i = 0; i < a.dimension(0); i++) {
    r = a.chip(i, 0);
    gauss1d(s, r, sy);
    a.chip(i, 0) = s;
  }
  for (int j = 0; j < a.dimension(1); j++) {
    r = a.chip(j, 1);
    gauss1d(s, r, sx);
    a.chip(j, 1) = s;
  }
}

void gauss2d(TensorMap2 image, float sx, float sy);

inline int clip(int x, int n) {
  if (x < 0) return 0;
  if (x >= n) return n - 1;
  return x;
}

inline Float bilin(TensorMap2 a, float x, float y) {
  int w = a.dimension(0), h = a.dimension(1);
  int i = (int)floor(x);
  int j = (int)floor(y);
  float l = x - i;
  float m = y - j;
  float s00 = a(clip(i, w), clip(j, h));
  float s01 = a(clip(i, w), clip(j + 1, h));
  float s10 = a(clip(i + 1, w), clip(j, h));
  float s11 = a(clip(i + 1, w), clip(j + 1, h));
  return ((1.0 - l) * ((1.0 - m) * s00 + m * s01) +
          l * ((1.0 - m) * s10 + m * s11));
}

struct NoNormalizer : INormalizer {
  void measure(TensorMap2 line) {}
  void normalize(Tensor2 &out, TensorMap2 in) {
    assert(in.dimension(1) == target_height);
    out = in;
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
  void measure(TensorMap2 line) {
    {
      double sy = 0, s1 = 0;
      for (int i = 0; i < line.dimension(0); i++) {
        for (int j = 0; j < line.dimension(1); j++) {
          sy += line(i, j) * j;
          s1 += line(i, j);
        }
      }
      y_mean = sy / s1;
    }
    {
      double sy = 0, s1 = 0;
      for (int i = 0; i < line.dimension(0); i++) {
        for (int j = 0; j < line.dimension(1); j++) {
          sy += line(i, j) * fabs(j - y_mean);
          s1 += line(i, j);
        }
      }
      y_mad = sy / s1;
    }
  }
  void normalize(Tensor2 &out, TensorMap2 in) {
    float actual = vscale * 2 * range * y_mad;
    float scale = actual / target_height;
    cerr << "normalize: " << y_mean << " " << y_mad << " " << actual << endl;
    int nw = int(in.dimension(0) / scale);
    int nh = target_height;
    out.resize(nw, nh);
    for (int i = 0; i < nw; i++) {
      for (int j = 0; j < nh; j++) {
        float x = scale * i;
        float y = scale * (j - target_height / 2) + y_mean;
        out(i, j) = bilin(in, x, y);
      }
    }
  }
};

void argmax1(Tensor<float, 1> &m, TensorMap2 a) {
  m.resize(a.dimension(0));
  for (int i = 0; i < a.dimension(0); i++) {
    float mv = a(i, 0);
    float mj = 0;
    for (int j = 1; j < a.dimension(1); j++) {
      if (a(i, j) < mv) continue;
      mv = a(i, j);
      mj = j;
    }
    m(i) = mj;
  }
}

inline void add_smear(TensorMap2 smooth, TensorMap2 line) {
  int w = line.dimension(0);
  int h = line.dimension(1);
  for (int j = 0; j < h; j++) {
    double v = 0.0;
    for (int i = 0; i < w; i++) {
      v = v * 0.9 + line(i, j);
      smooth(i, j) += fmin(1.0, v) * 1e-3;
    }
  }
}

struct CenterNormalizer : INormalizer {
  pytensor::PyServer *py = 0;
  Tensor<float, 1> center;
  float r = -1;
  void setPyServer(void *p) { this->py = (pytensor::PyServer *)p; }
  void getparams(bool verbose) {
    range = getrenv("norm_range", 4.0);
    smooth2d = getrenv("norm_smooth2d", 1.0);
    smooth1d = getrenv("norm_smooth1d", 0.3);
    if (verbose) print("center_normalizer", range, smooth2d, smooth1d);
  }
  void measure(TensorMap2 line) {
    Tensor2 smooth, smooth2;
    int w = line.dimension(0);
    int h = line.dimension(1);
    smooth = line;
    gauss2d(smooth(), h * smooth2d, h * 0.5);
    add_smear(smooth(), line);  // just to avoid singularities
    Tensor<float, 1> a(w);
    argmax1(a, smooth());
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
#ifdef FIXME
      print("r", r);
      py->eval("ion(); clf()");
      py->eval("subplot(211)");
      py->imshowT(line, "cmap=cm.gray,interpolation='nearest'");
      py->eval("subplot(212)");
      py->imshowT(smooth, "cmap=cm.gray,interpolation='nearest'");
      py->plot(center);
      py->eval("print ginput(999)");
#endif
    }
  }
  void normalize(Tensor2 &out, TensorMap2 in) {
    int w = in.dimension(0);
    if (w != center.dimension(0)) THROW("measure doesn't match normalize");
    float scale = (2.0 * r) / target_height;
    int target_width = max(int(w / scale), 1);
    out.resize(target_width, target_height);
    for (int i = 0; i < out.dimension(0); i++) {
      for (int j = 0; j < out.dimension(1); j++) {
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
  return 0;
}

// PNG I/O (taken from iulib)

#define CHECK_CONDITION(X)         \
  do {                             \
    if (!(X)) THROW("CHECK: " #X); \
  } while (0)
#define CHECK_ARG(X)                   \
  do {                                 \
    if (!(X)) THROW("CHECK_ARG: " #X); \
  } while (0)

bool png_flip = false;

void read_png(Tensor<unsigned char, 3> &image, FILE *fp) {
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

  if (!fp) THROW("fp not defined");

  // Allocate the 3 data structures
  if ((png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
                                        NULL, NULL)) == NULL)
    THROW("png_ptr not made");

  if ((info_ptr = png_create_info_struct(png_ptr)) == NULL) {
    png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    THROW("info_ptr not made");
  }

  if ((end_info = png_create_info_struct(png_ptr)) == NULL) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    THROW("end_info not made");
  }

  // Set up png setjmp error handling

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    THROW("internal png error");
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
    THROW("there shouldn't be 2 spp!");
  } else if (spp == 3) {
    d = 4 * bit_depth;
  } else { /* spp == 4 */
    d = 4 * bit_depth;
    THROW("there shouldn't be 4 spp!");
  }

  /* Remove if/when this is implemented for all bit_depths */
  if (spp == 3 && bit_depth != 8) {
    fprintf(stderr, "Help: spp = 3 and depth = %d != 8\n!!", bit_depth);
    THROW("not implemented for this depth");
  }

  Tensor<int, 2> color_map;

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

  assert(w > 0 && h > 0);
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
        image(x, y, 0) = value;
        image(x, y, 1) = value;
        image(x, y, 2) = value;
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
        image(x, y, 0) = rowptr[k++];
        image(x, y, 1) = rowptr[k++];
        image(x, y, 2) = rowptr[k++];
      }
    }
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
}

void write_png(FILE *fp, Tensor<unsigned char, 3> &image) {
  int d;
  png_byte bit_depth, color_type;
  int w, h;
  png_structp png_ptr;
  png_infop info_ptr;
  unsigned int default_xres = 300;
  unsigned int default_yres = 300;

  CHECK_ARG(image.rank() == 2 ||
            (image.rank() == 3 && image.dimension(2) == 3));

  if (!fp) THROW("stream not open");

  /* Allocate the 2 data structures */
  if ((png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
                                         NULL, NULL)) == NULL)
    THROW("png_ptr not made");

  if ((info_ptr = png_create_info_struct(png_ptr)) == NULL) {
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    THROW("info_ptr not made");
  }

  /* Set up png setjmp error handling */
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    THROW("internal png error");
  }

  png_init_io(png_ptr, fp);

  w = image.dimension(0);
  h = image.dimension(1);
  d = image.dimension(2);
  bit_depth = 8;
  color_type = PNG_COLOR_TYPE_RGB;

  png_set_IHDR(png_ptr, info_ptr, w, h, bit_depth, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);
  png_set_pHYs(png_ptr, info_ptr, default_xres, default_yres,
               PNG_RESOLUTION_METER);
  png_write_info(png_ptr, info_ptr);

  Tensor<unsigned char, 1> rowbuffer;
  rowbuffer.resize(3 * w);
  for (int i = 0; i < h; i++) {
    int k = 0;
    for (int j = 0; j < w; j++) {
      int x = j;
      int y = png_flip ? (h - i - 1) : i;
      if (d == 1) {
        int value = image(x, y, 0);
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

inline double clip(double value, double lo, double hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

void read_png(Tensor2 &image, const char *name) {
  Tensor<unsigned char, 3> temp;
  FILE *stream = fopen(name, "r");
  if (!stream) THROW("error on open");
  read_png(temp, stream);
  fclose(stream);
  image.resize(temp.dimension(0), temp.dimension(1));
  for (int i = 0; i < temp.dimension(0); i++) {
    for (int j = 0; j < temp.dimension(1); j++) {
      if (temp.dimension(2) == 1)
        image(i, j) = temp(i, j, 0);
      else
        image(i, j) =
            (temp(i, j, 0) + temp(i, j, 1) + temp(i, j, 2)) / (3 * 255.0);
    }
  }
}
void write_png(const char *name, TensorMap2 image) {
  Tensor<unsigned char, 3> temp;
  temp.resize(image.dimension(0), image.dimension(1), 3);
  for (int i = 0; i < temp.dimension(0); i++) {
    for (int j = 0; j < temp.dimension(1); j++) {
      unsigned char value = floor(clip(image(i, j) * 256, 0.0, 255.999999));
      temp(i, j, 0) = value;
      temp(i, j, 1) = value;
      temp(i, j, 2) = value;
    }
  }
  FILE *stream = fopen(name, "w");
  if (!stream) THROW("error on open");
  write_png(stream, temp);
  fclose(stream);
}
}
