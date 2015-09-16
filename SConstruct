# -*- Python -*-
import os,sys,os.path
import distutils.sysconfig

if os.path.isdir(".hg"):
    hgversion = os.popen("hg -q id").read().strip()
elif os.path.isdir(".git"):
    hgversion = os.popen("git rev-list HEAD | sed 1q").read().strip()
else:
    hgversion = os.popen("date").read().strip()
print "version",hgversion

# A bunch of utility functions to make the rest of the SConstruct file a little simpler.

def die(msg):
    sys.stderr.write("ERROR "+msg+"\n")
    Exit(1)

def option(name,dflt):
    return (ARGUMENTS.get(name) or os.environ.get(name,dflt))

def findonpath(fname,path):
    for dir in path:
        if os.path.exists(os.path.join(dir,fname)):
            return dir
    raise die("%s: not found" % fname)

def protoc(target, source, env):
    os.system("protoc %s --cpp_out=." % source[0])
def protoemitter(target, source, env):
    for s in source:
        base,_ = os.path.splitext(str(s))
        target.extend([base+".pb.cc", base+".pb.h"])
    return target, source

protoc_builder = Builder(action=protoc,
                         emitter=protoemitter,
                         src_suffix=".proto")

# CLSTM requires C++11, and installes in /usr/local by default

prefix = option('prefix', "/usr/local")

env = Environment()
env.Append(CPPDEFINES={"HGVERSION" : '\\"'+hgversion+'\\"'})
env.Append(CPPDEFINES={'THROW' : 'throw'})
env["BUILDERS"]["Protoc"] = protoc_builder

# With omp=1 support, Eigen and other parts of the code may use multi-threading.

if option("omp",0):
    env["CXX"] = option("CXX", "g++") + " --std=c++11 -Wno-unused-result -fopenmp"
else:
    env["CXX"] = option("CXX", "g++") + " --std=c++11 -Wno-unused-result"

# With profile=1, the code will be compiled suitable for profiling and debug.
# With debug=1, the code will be compiled suitable for debugging.

if option("profile",0):
    env.Append(CXXFLAGS="-g -pg -fno-inline".split())
    env.Append(CCFLAGS="-g -pg".split())
    env.Append(LINKFLAGS="-g -pg".split())
elif option("debug",0):
    env.Append(CXXFLAGS="-g -fno-inline".split())
    env.Append(CCFLAGS="-g".split())
    env.Append(LINKFLAGS="-g".split())
else:
    env.Append(CXXFLAGS="-g -O3 -finline".split())
    env.Append(CCFLAGS="-g".split())

# Extra layers (old layers or testing)

if option("extras",0):
    env.Append(CPPDEFINES={'CLSTM_EXTRAS' : 1})

# Try to locate the Eigen include files (they are in different locations
# on different systems); you can specify an include path for Eigen with
# `eigen=/mypath/include`

if option("eigen","")=="":
    inc = findonpath("Eigen/Eigen","""
        /usr/include
        /usr/local/include/eigen3
        /usr/include/eigen3""".split())
else:
    inc = findonpath("Eigen/Eigen",[option("eigen")])
env.Append(CPPPATH=[inc])

# You can enable display debugging with `display=1`

if option("display",0):
    env.Append(LIBS=["zmqpp","zmq"])
    env.Append(CPPDEFINES={'add_raw' : option("add_raw",'add')})
else:
    env.Append(CPPDEFINES={'NODISPLAY' : 1})

env.Append(LIBS=["png","protobuf"])

# We need to compile the protocol buffer definition as part of the build.

env.Protoc("clstm.proto")

# Build the CLSTM library.

libs = env["LIBS"]
libsrc = ["clstm.cc", "ctc.cc", "clstm_proto.cc", "clstm_prefab.cc",
          "extras.cc", "clstm.pb.cc"]
libclstm = env.StaticLibrary("clstm", source = libsrc)

programs = """clstmtext clstmfilter clstmfiltertrain clstmocr clstmocrtrain""".split()
for program in programs:
    env.Program(program,[program+".cc"],LIBS=[libclstm]+libs)
    Default(program)

env.Program("test-forward",["test-forward.cc"],LIBS=[libclstm]+libs)

# env.Program("fstfun", "fstfun.cc", LIBS=[libclstm]+libs+["fst","dl"])

Alias('install-lib',
      Install(os.path.join(prefix,"lib"), libclstm))
Alias('install-include',
      Install(os.path.join(prefix,"include"), ["clstm.h"]))
Alias('install',
      ['install-lib', 'install-include'])

# If you have HDF5 installed, set hdf5lib=hdf5_serial (or something like that)
# and you will get a bunch of command line programs that can be trained from
# HDF5 data files. This code is messy and may get deprecated eventually.

if option("hdf5lib", "")!="":
    h5env = env.Clone()
    inc = findonpath("hdf5.h","""
        /usr/include
        /usr/local/include/hdf5/serial
        /usr/local/include/hdf5
        /usr/include/hdf5/serial
        /usr/include/hdf5""".split())
    h5env.Append(CPPPATH=[inc])
    h5env.Append(LIBS=["hdf5_cpp"])
    h5env.Append(LIBS=[option("hdf5lib", "hdf5_serial")])
    h5env.Prepend(LIBS=[libclstm])
    for program in "clstmctc clstmseq clstmconv".split():
        h5env.Program(program,[program+".cc"])


# You can construct the Python extension from scons using `pyswig=1`; however,
# the recommended way of compiling it is with "python setup.py build"

if option("pyswig", 0):
    swigenv = env.Clone( SWIGFLAGS=["-python","-c++"], SHLIBPREFIX="")
    swigenv.Append(CPPPATH=[distutils.sysconfig.get_python_inc()])
    swigenv.SharedLibrary("_clstm.so",
                          ["clstm.i", "clstm.cc", "extras.cc", "clstm.pb.cc"],
                          LIBS=libs)


destlib = distutils.sysconfig.get_config_var("DESTLIB")
Alias('pyinstall',
      Install(os.path.join(destlib, "site-packages"),
              ["_clstm.so", "clstm.py"]))
