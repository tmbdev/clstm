from cpython.ref cimport PyObject
from libc.stddef cimport wchar_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.string cimport string

cimport _clstm

from collections import namedtuple


CharPrediction = namedtuple("CharPrediction",
                            ("x_position", "char", "confidence"))


cdef extern from "Python.h":
    ctypedef PyObject PyUnicodeObject
    Py_ssize_t PyUnicode_AsWideChar(PyUnicodeObject *o, wchar_t *w,
                                    Py_ssize_t size)
    PyObject* PyUnicode_FromWideChar(wchar_t *w, Py_ssize_t size)


cpdef double levenshtein(unicode a, unicode b):
    return _clstm.levenshtein[string, string](
        a.encode('utf8'), b.encode('utf8'))


cdef load_img(img, _clstm.Tensor2 *data):
    """ Copy image data from a PIL image to an Eigen tensor.

    :param img:     Image
    :type img:      :py:class:`PIL.Image.Image`
    """
    data.resize(img.width, img.height)
    imgdata = img.load()
    for i in range(img.width):
        for j in range(img.height):
            px = imgdata[i, j]
            # Pillow returns pixels as [0, 255], but we need [0, 1]
            if isinstance(px, tuple):
                # For color images, we use the mean across all channels
                px = (sum(px)/len(px))/255.
            else:
                px = px/255.
            px = -px + 1.
            data[0].put(px, i, j)


cdef load_nparray(npdata, _clstm.Tensor2 *data):
    """ Copy image data from a numpy array to an Eigen tensor.

    This assumes that all pixels are floating point numbers in the range [0, 1]
    where 0 is white and 1 is black.

    Currently this copies all data, but it should be possible to make Eigen
    use the numpy-allocated data.
    """
    data.resize(npdata.shape[0], npdata.shape[1])
    for i in range(npdata.shape[0]):
        for j in range(npdata.shape[1]):
            data[0].put(npdata[i,j], i, j)


cdef class ClstmOcr:
    cdef _clstm.CLSTMOCR *_ocr

    def __cinit__(self, str fname=None):
        self._ocr = new _clstm.CLSTMOCR()
        if fname:
            self.load(fname)

    cpdef load(self, str fname):
        cdef bint rv = self._ocr.maybe_load(fname)
        if not rv:
            raise IOError("Could not load model from {}".format(fname))

    cpdef save(self, str fname):
        cdef bint rv = self._ocr.maybe_save(fname)
        if not rv:
            raise IOError("Could not save model to {}".format(fname))

    cpdef create_bidi(self, chars, int num_hidden):
        chars_str = u"".join(sorted(chars))
        cdef vector[int] codec
        cdef Py_ssize_t length = len(chars_str.encode("UTF-16")) // 2
        cdef wchar_t *wchars = <wchar_t *>malloc(length * sizeof(wchar_t))
        cdef Py_ssize_t number_written = PyUnicode_AsWideChar(
            <PyUnicodeObject *>chars_str, wchars, length)
        codec.push_back(0)
        for i in range(length-1):
            codec.push_back(<int>(wchars[i]))
        self._ocr.createBidi(codec, num_hidden)

    cpdef set_learning_rate(self, float learning_rate, float momentum):
        self._ocr.setLearningRate(learning_rate, momentum)

    def aligned(self):
        return self._ocr.aligned_utf8().decode('utf8')

    def train(self, img, unicode text):
        cdef _clstm.Tensor2 data
        if hasattr(img, 'width'):
            load_img(img, &data)
        elif hasattr(img, 'shape'):
            load_nparray(img, &data)
        return self._ocr.train_utf8(
            data.map(), text.encode('utf8')).decode('utf8')

    def recognize(self, img):
        cdef _clstm.Tensor2 data
        if hasattr(img, 'width'):
            load_img(img, &data)
        elif hasattr(img, 'shape'):
            load_nparray(img, &data)
        return self._ocr.predict_utf8(data.map()).decode('utf8')

    def recognize_chars(self, img):
        cdef _clstm.Tensor2 data
        cdef vector[_clstm.CharPrediction] preds
        cdef vector[_clstm.CharPrediction].iterator pred_it
        cdef wchar_t[2] cur_char
        if hasattr(img, 'width'):
            load_img(img, &data)
        elif hasattr(img, 'shape'):
            load_nparray(img, &data)
        self._ocr.predict(preds, data.map())
        for i in range(preds.size()):
            cur_char[0] = preds[i].c
            yield CharPrediction(
                preds[i].x,
                <unicode>PyUnicode_FromWideChar(cur_char, 1),
                preds[i].p)

    def __dealloc__(self):
        del self._ocr
