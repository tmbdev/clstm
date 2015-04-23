# clstm

[![Join the chat at https://gitter.im/tmbdev/clstm](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tmbdev/clstm?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A small C++ implementation of LSTM networks, focused on OCR.
The only essential external dependencies are Eigen and STL.

For I/O, having HDF5 is useful. For plotting, you need Python and the
ZMQ library.

To build a standalone C library, run

    scons
    sudo scons install

For debugging, you can compile with `debug=1`

To build the Python extension, run

    python setup.py build
    sudo python setup.py install

# C++ API

The `clstm` library operates on the Sequence type as its fundamental
data type, representing variable length sequences of fixed length vectors.
Internally, this is represented as an STL vector of Eigen dynamic vectors.

    typedef stl::vector<Eigen::VectorXf> Sequence;

Networks are built from objects implementing the `INetwork` interface.
The `INetwork` interface contains:

    struct INetwork {
        Sequence inputs, d_inputs;      // input sequence, input deltas
        Sequence outputs, d_outputs;    // output sequence, output deltas
        void forward();                 // propagate inputs to outputs
        void backward();                // propagate d_outputs to d_inputs
        void update();                  // update weights from the last backward() step
        void setLearningRate(Float,Float); // set learning rates
        ...
    };

Network structures can be hierarchical and there are some network 
implementations whose purpose it is to combine other networks into more
complex structures.

    struct INetwork {
        ...
        vector<shared_ptr<INetwork>> sub;
        void add(shared_ptr<INetwork> net);
        ...
    };

The most important of these is the `Stacked` network, which simply
stacks the given set of networks on top of each other, using the ouput
from each network as the input to the next. 

There are a few utility functions for walking through the subnetworks, states,
and weights of a network, together with two hooks (`preSave`, `postLoad`) to
facilitate loading.

The implementations of the various networks are not exposed; instead of
`new Stacked()` use `make_Stacked()`.

In addition to these basic functions, there is also a small implementation
of CTC alignment.

The C++ code roughly follows the lstm.py implementation from the Python version
of OCRopus. Gradients have been verified for the core LSTM implementation,
although there may be still be bugs in other parts of the code.

There is also a small multidimensional array class in `multidim.h`; that isn't
used in the core LSTM implementation, but it is used in debugging and testing
code, for plotting, and for HDF5 input/output. Unlike Eigen, it uses standard
C/C++ row major element order, as libraries like HDF5 expect.

LSTM models are stored by default in HDF5 models, using the `weights`
method to walk through all the weights of a network as arrays and storing
them in HDF5 arrays. If HDF5 doesn't suit your needs, you can write
similar functions for other forms of loading/saving LSTM networks.

# Python API

The `clstm.i` file implements a simple Python interface to clstm, plus
a wrapper that makes an INetwork mostly a replacement for the lstm.py
implementation from ocropy.

# Comand Line Drivers

There are several command line drivers:

  - clstmseq learns sequence-to-sequence mappings
  - clstmctc learns sequence-to-string mappings using CTC alignment
  - clstmtext learns string-to-string transformations

Note that most parameters are passed through the environment:

    lrate=3e-5 clstmctc uw3-dew.h5

Parameters are:

    clstmctc:

        Learns image->text transformations, modeling sequences of
        vertical slices through the input image.

        maxeval= max # evaluation samples
        randseed= random seed
        load= model to preload
        save_every= how often to save (0: save only improved models)
        after_save= shell command to execute after saving
        ntrain= number of training samples
        lrate= learning rate
        nhidden= #hidden units
        nhidden2= #hidden units second lstm layer
        batch= batching for updates
        momentum= momentum
        display_every= how often to display results
        report_every= how often to report progress
        randomize= shuffle training examples
        lrnorm= learning rate normalization
        dewarp= image dewarping method
        lstm= kind of LSTM to be used
        testset= test set file
        test_every= how often to compute test error rate
        after_test= shell command to run after testing
        start= start sample for training
        mode= command line mode (training, errors, etc.)

    clstmseq:

        lrate= learning rate
        display_every= how often to display recognition output (0=never)
        report= how often to report progress
        ntrain= total number of training steps
        kind= bidi, bidi2, lstm1
        state= number of internal state variables (2 by default)

    clstmtext:

        Training files contain lines of the form: "input\toutput\n"
        With mode=filter, the input file is transformed using the
        trained filter.

        (similar to clstmctc)

(You can find all parameters via `grep 'get.env' *.cc`.)

For debugging and testing, there are equivalent Python implementations
(`pylstmseq` and `pylstmctc`) that should work the same way in Python.

The `rnntests.py` script will generate a number of simple sequence
recognition tasks for testing.

# TODO / UPCOMING

  - the HDF5 network save format will probably change
  - Lua and Torch bindings
  - 2D LSTM support
