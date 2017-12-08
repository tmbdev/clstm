# -*- Python -*-

# CLSTM requires C++11, and installs in /usr/local by default

import os
import sys
import os.path
import distutils.sysconfig

# A bunch of utility functions to make the rest of the SConstruct file a
# little simpler.

def die(msg):
    sys.stderr.write("ERROR " + msg + "\n")
    Exit(1)
def option(name, dflt):
    result = (ARGUMENTS.get(name) or os.environ.get(name, dflt))
    if type(dflt)==int: result = int(result)
    return result
def findonpath(fname, path):
    for dir in path:
        if os.path.exists(os.path.join(dir, fname)):
            return dir
    die("%s: not found" % fname)

# A protocol buffer builder.

def protoc(target, source, env):
    os.system("protoc %s --cpp_out=." % source[0])
def protoemitter(target, source, env):
    for s in source:
        base, _ = os.path.splitext(str(s))
        target.extend([base + ".pb.cc", base + ".pb.h"])
    return target, source

protoc_builder = Builder(action=protoc,
                         emitter=protoemitter,
                         src_suffix=".proto")

prefix = option('prefix', "/usr/local")

env = Environment()
env.Append(CPPDEFINES={'THROW': 'throw', 'CATCH': 'catch', 'TRY': 'try'})
env.Append(CPPDEFINES={'CLSTM_ALL_TENSOR': '1'})
env["BUILDERS"]["Protoc"] = protoc_builder

options = option("options", "")
env["CXX"] = option("CXX", "g++") + " --std=c++11 -Wno-unused-result "+options

if option("double", 0):
    env.Append(CPPDEFINES={'LSTM_DOUBLE': '1'})


# With profile=1, the code will be compiled suitable for profiling and debug.
# With debug=1, the code will be compiled suitable for debugging.

profile = option("profile", 0)
debug = option("debug", 0)

if profile>0:
    #env.Append(CXXFLAGS="-g -pg -O2".split())
    env.Append(CCFLAGS="-g -pg -O2".split())
    env.Append(LINKFLAGS="-g -pg".split())
elif debug>1:
    #env.Append(CXXFLAGS="-g -fno-inline".split())
    env.Append(CCFLAGS="-g".split())
    env.Append(LINKFLAGS="-g".split())
elif debug>0:
    #env.Append(CXXFLAGS="-g".split())
    env.Append(CCFLAGS="-g".split())
    env.Append(LINKFLAGS="-g".split())
elif debug==0:
    #env.Append(CXXFLAGS="-g -O3 -DEIGEN_NO_DEBUG".split())
    env.Append(CCFLAGS="-g -O3 -DEIGEN_NO_DEBUG".split())
elif debug<0:
    env.Append(CCFLAGS="-g -Ofast -DEIGEN_NO_DEBUG -finline -ffast-math -fno-signaling-nans -funsafe-math-optimizations -ffinite-math-only -march=native".split())

# Try to locate the Eigen include files (they are in different locations
# on different systems); you can specify an include path for Eigen with
# `eigen=/mypath/include`

if option("eigen", "") == "":
    inc = findonpath("Eigen/Eigen", """
        /usr/local/include
        /usr/local/include/eigen3
        /usr/include
        /usr/include/eigen3""".split())
else:
    inc = findonpath("Eigen/Eigen", [option("eigen", "")])

env.Append(CPPPATH=[inc])
env.Append(LIBS=["png", "protobuf"])

# You can enable display debugging with `display=1` (probably not working right now)

if option("display", 0):
    env.Append(LIBS=["zmqpp", "zmq"])
    env.Append(CPPDEFINES={'add_raw': option("add_raw", 'add')})
else:
    env.Append(CPPDEFINES={'NODISPLAY': 1})

if option("openmp", 0):
    env.Append(CCFLAGS="-fopenmp")

# We need to compile the protocol buffer definition as part of the build.

env.Protoc("clstm.proto")

cuda = env.Object("clstm_compute_cuda.o", "clstm_compute_cuda.cc",
           CXX="./nvcc-wrapper")

# Build the CLSTM library.

libsrc = ["clstm.cc", "ctc.cc", "clstm_proto.cc", "clstm_prefab.cc",
          "tensor.cc", "batches.cc", "extras.cc", "clstm.pb.cc", 
          "clstm_compute.cc"]
if option("gpu", 0):
  env.Append(LIBS=["cudart","cublas","cuda"])
  env.Append(LIBPATH=["/usr/local/cuda/lib64"])
  env.Append(CPPPATH=["/usr/local/cuda/include"])
  env.Append(CPPDEFINES={'CLSTM_CUDA' : 1, 'EIGEN_USE_GPU' : 1})
  libsrc = [cuda] + libsrc

libs = env["LIBS"]
libclstm = env.StaticLibrary("clstm", libsrc)

all = [libclstm]

programs = """clstmfilter clstmfiltertrain clstmocr clstmocrtrain""".split()
for program in programs:
    all += [env.Program(program, [program + ".cc"], LIBS=[libclstm] + libs)]
    Default(program)

# env.Program("fstfun", "fstfun.cc", LIBS=[libclstm]+libs+["fst","dl"])

Alias('install-lib',
      Install(os.path.join(prefix, "lib"), libclstm))
Alias('install-include',
      Install(os.path.join(prefix, "include"), ["clstm.h"]))
Alias('install',
      ['install-lib', 'install-include'])

# A simple test of the C++ LSTM implementation.
all += [env.Program("test-lstm", ["test-lstm.cc"], LIBS=[libclstm] + libs)]
all += [env.Program("test-lstm2", ["test-lstm2.cc"], LIBS=[libclstm] + libs)]
all += [env.Program("test-batchlstm", ["test-batchlstm.cc"], LIBS=[libclstm] + libs)]
all += [env.Program("test-deriv", ["test-deriv.cc"], LIBS=[libclstm] + libs)]
all += [env.Program("test-cderiv", ["test-cderiv.cc"], LIBS=[libclstm] + libs)]
all += [env.Program("test-ctc", ["test-ctc.cc"], LIBS=[libclstm] + libs)]
all += [env.Program("test-2d", ["test-2d.cc"], LIBS=[libclstm] + libs)]

# You can construct the Python extension from scons using the `pyswig` target; however,
# the recommended way of compiling it is with "python setup.py build"

swigenv = env.Clone()
swigenv.Tool("swig")
swigenv.Append(SWIG="3.0")
swigenv.Append(CPPPATH=[distutils.sysconfig.get_python_inc()])
pyswig = swigenv.SharedLibrary("_clstm.so",
                               ["clstm.i", "clstm.cc", "clstm_proto.cc", "extras.cc",
                                "clstm.pb.cc", "clstm_compute.cc",
                               "clstm_prefab.cc", "ctc.cc"],
                               SWIGFLAGS=['-python', '-c++'],
                               SHLIBPREFIX="",
                               LIBS=libs)
Alias('pyswig', [pyswig])

destlib = distutils.sysconfig.get_config_var("DESTLIB")
Alias('pyinstall',
      Install(os.path.join(destlib, "site-packages"),
              ["_clstm.so", "clstm.py"]))

Alias('all', [all])
