from cpython.ref cimport PyObject
from libc.stddef cimport wchar_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr


cdef extern from "<string>" namespace "std":
    cppclass wstring:
        cppclass iterator:
            iterator()
            wchar_t* operator*()
            iterator(iterator &)
            iterator operator++()
            iterator operator--()
            iterator operator==(iterator)
            iterator operator!=(iterator)
        iterator begin()
        iterator end()


cdef extern from "pyextra_defs.h":
    cdef Py_ssize_t Unicode_AsWideChar(PyObject* ustr, Py_ssize_t length,
                                       wchar_t* wchars)


cdef extern from "pstring.h":
    wstring utf8_to_utf32(string s)


cdef extern from "clstm.h":
    cdef double levenshtein[A, B](A a, B b)


cdef extern from "clstm.h" namespace "ocropus":
    cdef cppclass Assoc:
        Assoc()
        Assoc(string &s)
        bint contains(string &key, bint parent = true)
        string get(string &key)
        string get(string &key, string default)
        void set(string &key, string value)

    cdef cppclass INetwork:
        Assoc attr

    ctypedef shared_ptr[INetwork] Network


cdef extern from "tensor.h" namespace "ocropus":
    cppclass TensorMap2:
        pass

    cdef cppclass Tensor2:
        int dims[2]
        float *ptr
        void resize(int i, int j)
        void put(float val, int i, int j)
        float get(int i, int j)
        TensorMap2 map()

cdef extern from "clstmhl.h" namespace "ocropus":
    struct CharPrediction:
        int i
        int x
        wchar_t c
        float p

    # NOTE: The content of `codec` should be the utf-32 characters that the
    #       network is supposed to learn, encoded as integers
    cppclass CLSTMOCR:
        int target_height
        Network net
        bint maybe_load(string &fname)
        bint maybe_save(string &fname)
        void createBidi(vector[int] codec, int nhidden)
        void setLearningRate(float learning_rate, float momentum)
        string train_utf8(TensorMap2 imgdata, string &target)
        string predict_utf8(TensorMap2 imgdata)
        void predict(vector[CharPrediction] &preds, TensorMap2 imgdata)
        string aligned_utf8()
