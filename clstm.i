// -*- C++ -*-

%{
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
%}

%module(docstring="C-version of the ocropy LSTM implementation") clstm;
%feature("autodoc",1);
%include "typemaps.i"
%include "std_string.i"
%include "std_wstring.i"
%include "std_shared_ptr.i"
%include "std_vector.i"
%shared_ptr(ITrainable)
%shared_ptr(INetwork)
#ifdef SWIGPYTHON
%include "cstring.i"
#endif

%{
#include <memory>
#include <iostream>
#include "clstm.h"
#include "clstm_compute.h"
using namespace ocropus;
using namespace std;
%}

typedef float Float;
using std::string;

#ifdef SWIGPYTHON
%exception {
    try {
        $action
    }
    catch(const char *s) {
        PyErr_SetString(PyExc_IndexError,s);
        return NULL;
    }
    catch(...) {
        PyErr_SetString(PyExc_IndexError,"unknown exception in iulib");
        return NULL;
    }
}
#endif

%{
#include "numpy/arrayobject.h"
%}

%init %{
import_array();
%}

/* create simple interface definitions for the built-in Sequence types */

struct Classes {
    Classes();
    ~Classes();
    %rename(__getitem__) operator[];
    int operator[](int i);
    int size();
    void resize(int);
};
%extend Classes {
    void __setitem__(int i,int value) {
        (*$self)[i] = value;
    }
}

struct Batch {
  void clear();
  int rows();
  int cols();
  float &v(int,int);
  float &d(int,int);
};

struct Params {
  void resize(int,int);
  void setZero(int,int);
  int rows();
  int cols();
  float &v(int,int);
  float &d(int,int);
};


struct Sequence {
    Sequence();
    ~Sequence();
    int size();
    int rows();
    int cols();
    %rename(__getitem__) operator[];
    Batch &operator[](int i);
};

struct Assoc {
  string get(string key);
  string get(string key, string dflt);
  void set(string key, string dflt);
};

struct Codec {
  std::vector<int> codec;
  int size() { return codec.size(); }
  void set(const vector<int> &data);
  wchar_t decode(int cls);
  std::wstring decode(Classes &cs);
  void encode(Classes &cs, const std::wstring &s);
private:
  void operator=(const Codec &);
};

struct INetwork;
typedef std::shared_ptr<INetwork> Network;
%template(vectornet) std::vector<std::shared_ptr<INetwork> >;

struct INetwork {
    string kind;
    Assoc attr;
    virtual void setLearningRate(Float lr, Float momentum) = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void initialize();
    virtual ~INetwork();
    Sequence inputs;
    Sequence outputs;
    std::vector<std::shared_ptr<INetwork> > sub;
    Codec codec;
    Codec icodec;
    virtual int ninput();
    virtual int noutput();
    virtual void add(std::shared_ptr<INetwork> net);
};

void sgd_update(Network net);
void set_inputs(Network net, Sequence &inputs);
void set_targets(Network net, Sequence &targets);
void set_classes(Network net, Classes &classes);
void mktargets(Sequence &seq, Classes &targets, int ndim);

std::shared_ptr<INetwork> make_layer(string);
std::shared_ptr<INetwork> make_net_init(string,string);

#if 0
%rename(seq_forward) forward_algorithm;
void forward_algorithm(Mat &lr,Mat &lmatch,double skip=-5.0);
%rename(seq_forwardbackward) forwardbackward;
void forwardbackward(Mat &both,Mat &lmatch);
#endif

%rename(seq_ctc_align) ctc_align_targets;
void ctc_align_targets(Sequence &posteriors,Sequence &outputs,Sequence &targets);
void mktargets(Sequence &seq, Classes &targets, int ndim);

void save_net(const string &file, Network net);
Network load_net(const string &file);

%inline %{
int string_edit_distance(string a, string b) {
    return levenshtein(a, b);
}

string network_info(Network net) {
    string result = "";
    walk_networks(net, [&result] (string s, INetwork *net) {
        double lr = net->attr.get("learning_rate","-1");
        double momentum = net->attr.get("momentum","-1");
        result += s + ": " + to_string(lr);
        result += string(" ") + to_string(momentum);
        result += string(" ") + to_string(net->ninput());
        result += string(" ") + to_string(net->noutput());
        result += "\n";
    });
    return result;
}

string sequence_info(Sequence &seq) {
    string result = "";
    result += to_string(seq.size());
    result += string(":") + (seq.size()>0?to_string(seq[0].rows()):"*");
    result += string(":") + (seq.size()>0?to_string(seq[0].cols()):"*");
#if 0
    // FIXME
    double lo = 1e99, hi = -1e99;
    for (int t=0;t<seq.size(); t++) {
        lo = fmin(lo, minimum(seq[t].V()));
        hi = fmax(hi, maximum(seq[t].V()));
    }
    result += "[" + to_string(lo) + "," + to_string(hi) + "]";
#endif
    return result;
}

%}

#ifdef SWIGPYTHON
%{
#include "numpyarray.h"
%}
%inline %{
void sequence_of_array(Sequence &a,PyObject *object_) {
    npa_float np(object_);
    if(np.rank()!=3) throw "rank must be 3";
    int N = np.dim(0);
    int d = np.dim(1);
    int bs = np.dim(2);
    a.resize(N,d,bs);
    for(int t=0;t<N;t++) {
        for(int i=0; i<d; i++)
            for(int b=0; b<bs; b++)
                a[t].v(i,b) = np(t,i,b);
    }
}

void d_sequence_of_array(Sequence &a,PyObject *object_) {
    npa_float np(object_);
    if(np.rank()!=3) throw "rank must be 3";
    int N = np.dim(0);
    int d = np.dim(1);
    int bs = np.dim(2);
    if (a.size() != N) throw "size mismatch";
    for(int t=0;t<N;t++) {
        for(int i=0; i<d; i++)
            for(int b=0; b<bs; b++)
                a[t].d(i,b) = np(t,i,b);
    }
}

void array_of_sequence(PyObject *object_,Sequence &a) {
    npa_float np(object_);
    int N = a.size();
    if (N==0) throw "empty sequence";
    int d = a[0].rows();
    if (d==0) throw "empty feature vector";
    int bs = a[0].cols();
    if (bs==0) throw "empty batch";
    np.resize(N,d,bs);
    for(int t=0; t<N; t++) {
        for(int i=0; i<d; i++)
            for(int b=0; b<bs; b++)
                np(t,i,b) = a[t].v(i,b);
    }
}

void array_of_d_sequence(PyObject *object_,Sequence &a) {
    npa_float np(object_);
    int N = a.size();
    if (N==0) throw "empty sequence";
    int d = a[0].rows();
    if (d==0) throw "empty feature vector";
    int bs = a[0].cols();
    if (bs==0) throw "empty batch";
    np.resize(N,d,bs);
    for(int t=0; t<N; t++) {
        for(int i=0; i<d; i++)
            for(int b=0; b<bs; b++)
                np(t,i,b) = a[t].d(i,b);
    }
}
%}

%pythoncode %{
import numpy

def Sequence_array(self):
    a = numpy.zeros(1,'f')
    array_of_sequence(a, self)
    return a
Sequence.array = Sequence_array

def Sequence_aset(self, a):
    sequence_of_array(self, a)
Sequence.aset = Sequence_aset

def Sequence_darray(self):
    a = numpy.zeros(1,'f')
    array_of_d_sequence(a, self)
    return a
Sequence.darray = Sequence_darray

def Sequence_dset(self, a):
    d_sequence_of_array(self, a)
Sequence.dset = Sequence_dset

def ctcalign(outputs_,targets_):
    outputs = Sequence()
    targets = Sequence()
    outputs.aset(outputs_)
    targets.aset(targets_)
    posteriors = Sequence()
    seq_ctc_align(posteriors,outputs,targets)
    return posteriors.array()
%}
#endif
