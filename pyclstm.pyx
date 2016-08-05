from cpython.ref cimport PyObject
from libc.stddef cimport wchar_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.set cimport set

cimport _clstm

cdef extern from "Python.h":
    ctypedef PyObject PyUnicodeObject
    Py_ssize_t PyUnicode_AsWideChar(PyUnicodeObject *o, wchar_t *w,
                                    Py_ssize_t size)


#cdef make_codec(texts):
#    cdef set[int] codes
#    cdef _clstm.wstring s
#    for t in texts:
#        s = _clstm.utf8_to_utf32(t.encode('utf8'))
#        for c in s:
##            codes.insert(<int>c)
#    cdef vector[int] codec
#    for c in codes:
#        codec.push_back(c)
#    return codes


cdef class ClstmOcr:
    cdef _clstm.CLSTMOCR *_ocr

    def __cinit__(self, str fname=None):
        self._ocr = new _clstm.CLSTMOCR()

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

    cpdef unicode train(self, img, unicode text):
        cdef _clstm.Tensor2 data
        data.resize(img.height, img.width)
        for i in range(img.height):
            for j in range(img.width):
                px = img.getpixel((j, i))
                if isinstance(px, tuple):
                    px = sum(px)/len(px)
                data.put(px, i, j)
        return self._ocr.train_utf8(
            data.map(), text.encode('utf8')).decode('utf8')

    def __dealloc__(self):
        del self._ocr
