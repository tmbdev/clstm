# clstm

[![CircleCI](https://circleci.com/gh/tmbdev/clstm/tree/master.svg?style=svg)](https://circleci.com/gh/tmbdev/clstm/tree/master)

CLSTM is an implementation of the
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) recurrent neural
network model in C++, using the [Eigen](http://eigen.tuxfamily.org) library for
numerical computations.

# Status and scope

CLSTM is mainly in maintenance mode now. It was created at a time when there weren't a lot of good LSTM
implementations around, but several good options have become available over the last year. Nevertheless, if
you need a small library for text line recognition with few dependencies, CLSTM is still a good option.

# Installation using Docker

You can train and run clstm without installation to the local machine using the
docker image, which is based on Ubuntu 16.04. This is the best option for
running clstm on a Windows host.

You can either run the [last version of the clstm
image](https://hub.docker.com/r/kbai/clstm) from Docker Hub or build the Docker
image from the repo (see [`./docker/Dockerfile`](./docker/Dockerfile)).

The command line syntax differs from a native installation:

```
docker run --rm -it -e [VARIABLES...] kbai/clstm BINARY [ARGS...]
```

is equivalent to

```
[VARIABLES...] BINARY [ARGS...]
```

For example:

```
docker run --rm -it -e ntrain=1000 kbai/clstm clstmocrtrain traininglist.txt
```

is equivalent to

```
ntrain=1000 clstmocrtrain traininglist.txt
```

# Installation from source

## Prerequisites

 - scons, swig, Eigen
 - protocol buffer library and compiler
 - libpng
 - Optional: HDF5, ZMQ, Python

```sh
# Ubuntu 15.04, 16.04 / Debian 8, 9
sudo apt-get install scons libprotobuf-dev protobuf-compiler libpng-dev libeigen3-dev swig

# Ubuntu 14.04:
sudo apt-get install scons libprotobuf-dev protobuf-compiler libpng-dev swig
```

The Debian repositories jessie-backports and stretch include sufficiently new libeigen3-dev packages.

It is also possible to download [Eigen](http://eigen.tuxfamily.org) with Tensor support (> v3.3-beta1)
and copy the header files to an `include` path:

```sh
# with wget
wget 'https://github.com/RLovelett/eigen/archive/3.3-rc1.tar.gz'
tar xf 3.3-rc1.tar.gz
rm -f /usr/local/include/eigen3
mv eigen-3.3-rc1 /usr/local/include/eigen3
# or with git:
sudo git clone --depth 1 --single-branch --branch 3.3-rc1 \
  "https://github.com/RLovelett/eigen" /usr/local/include/eigen3
```

To use the [visual debugging methods](#user-content-display), additionally:

```sh
# Ubuntu 15.04:
sudo apt-get install libzmq3-dev libzmq3 libzmqpp-dev libzmqpp3 libpng12-dev
```

For [HDF5](#user-content-hdf5), additionally:

```sh
# Ubuntu 15.04:
sudo apt-get install hdf5-helpers libhdf5-8 libhdf5-cpp-8 libhdf5-dev python-h5py

# Ubuntu 14.04:
sudo apt-get install hdf5-helpers libhdf5-7 libhdf5-dev python-h5py
```

## Building

To build a standalone C library, run

    scons
    sudo scons install

There are a bunch of options:

 - `debug=1` build with debugging options, no optimization
 - <a id="display">`display=1`</a> build with display support for debugging (requires ZMQ, Python)
 - `prefix=...` install under a different prefix (untested)
 - `eigen=...` where to look for Eigen include files (should contain `Eigen/Eigen`)
 - `openmp=...` build with multi-processing support. Set the
   [`OMP_NUM_THREADS`](https://eigen.tuxfamily.org/dox/TopicMultiThreading.html)
   environment variable to the number of threads for Eigen to use.
 - <a id="hdf5">`hdf5lib=hdf5`</a> what HDF5 library to use; enables HDF5 command line 
   programs (may need `hdf5_serial` in some environments)

## Running the tests

After building the executables, you can run two simple test runs as follows:

 - `run-cmu` will train an English-to-IPA LSTM
 - `run-uw3-500` will download a small OCR training/test set and train an OCR LSTM

There is a full set of tests in the current version of clstm; just
run them with:

```sh
./run-tests
```

This will check:

 - gradient checkers for layers and compute steps
 - training a simple model through the C++ API
 - training a simple model through the Python API
 - checking the command line training tools, including loading and saving

## Python bindings

To build the Python extension, run

```sh
python setup.py build
sudo python setup.py install
```

(this is currently broken)

# Documentation / Examples

You can find some documentation and examples in the form of iPython notebooks in the `misc` directory
(these are version 3 notebooks and won't open in older versions).

You can view these notebooks online here:
http://nbviewer.ipython.org/github/tmbdev/clstm/tree/master/misc/

# C++ API

The `clstm` library operates on the Sequence type as its fundamental
data type, representing variable length sequences of fixed length vectors.
The underlying Sequence type is a rank 4 tensor with accessors for
individual rank-2 tensors at different time steps.

Networks are built from objects implementing the `INetwork` interface.
The `INetwork` interface contains:

```c++
struct INetwork {
    Sequence inputs, d_inputs;      // input sequence, input deltas
    Sequence outputs, d_outputs;    // output sequence, output deltas
    void forward();                 // propagate inputs to outputs
    void backward();                // propagate d_outputs to d_inputs
    void update();                  // update weights from the last backward() step
    void setLearningRate(Float,Float); // set learning rates
    ...
};
```

Network structures can be hierarchical and there are some network
implementations whose purpose it is to combine other networks into more
complex structures.

```c++
struct INetwork {
    ...
    vector<shared_ptr<INetwork>> sub;
    void add(shared_ptr<INetwork> net);
    ...
};
```

At its lowest level, layers are created by:

 - create an instance of the layer with `make_layer`
 - set any parameters (including `ninput` and `noutput`) as
   attributes
 - add any sublayers to the `sub` vector
 - call `initialize()`

There are three different functions for constructing layers and networks:

 - `make_layer(kind)` looks up the constructor and gives you an uninitialized layer
 - `layer(kind,ninput,noutput,args,sub)` performs all initialization steps in sequence
 - `make_net(kind,args)` initializes a whole collection of layers at once
 - `make_net_init(kind,params)` is like `make_net`, but parameters are given in string form

The `layer(kind,ninput,noutput,args,sub)` function will perform
these steps in sequence.

Layers and networks are usually passed around as `shared_ptr<INetwork>`;
there is a typedef of this calling it `Network`.

This can be used to construct network architectures in C++ pretty
easily. For example, the following creates a network that stacks
a softmax output layer on top of a standard LSTM layer:

```c++
Network net = layer("Stacked", ninput, noutput, {}, {
    layer("LSTM", ninput, nhidden,{},{}),
    layer("SoftmaxLayer", nhidden, noutput,{},{})
});
```

Note that you need to make sure that the number of input and
output units are consistent between layers.

In addition to these basic functions, there is also a small implementation
of CTC alignment.

The C++ code roughly follows the lstm.py implementation from the Python
version of OCRopus. Gradients have been verified for the core LSTM
implementation, although there may be still be bugs in other parts of
the code.

There is also a small multidimensional array class in `multidim.h`; that
isn't used in the core LSTM implementation, but it is used in debugging
and testing code, for plotting, and for HDF5 input/output. Unlike Eigen,
it uses standard C/C++ row major element order, as libraries like
HDF5 expect. (NB: This will be replaced with Eigen::Tensor.)

LSTM models are stored in protocol buffer format (`clstm.proto`),
although adding new formats is easy. There is an older HDF5-based
storage format.

# Python API

The `clstm.i` file implements a simple Python interface to clstm, plus
a wrapper that makes an INetwork mostly a replacement for the lstm.py
implementation from ocropy.

# Command Line Drivers

There are several command line drivers:

  - `clstmfiltertrain training-data test-data` learns text filters;
    - input files consiste of lines of the form "input<tab>output<nl>"
  - `clstmfilter` applies learned text filters
  - `clstmocrtrain training-images test-images` learns OCR (or image-to-text) transformations;
    - input files are lists of text line images; the corresponding UTF-8 ground truth is expected in the corresponding `.gt.txt` file
  - `clstmocr` applies learned OCR models

 In addition, you get the following HDF5-based commands:

  - clstmseq learns sequence-to-sequence mappings
  - clstmctc learns sequence-to-string mappings using CTC alignment
  - clstmtext learns string-to-string transformations

Note that most parameters are passed through the environment:

```
lrate=3e-5 clstmctc uw3-dew.h5
```

See the notebooks in the `misc/` subdirectory for documentation on the parameters and examples of usage.

(You can find all parameters via `grep 'get.env' *.cc`.)
