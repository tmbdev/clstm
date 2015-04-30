# -*- Python -*-
import os
import distutils.sysconfig

def die(msg):
    sys.stderr.write("ERROR "+msg+"\n")
    Exit(1)

version = os.popen("hg -q id").read()[:-1]
hglog = os.popen("hg log --limit 3").read()
open("version.h","w").write("""
inline const char *hg_version() {
    return "%s";
}
inline const char *hg_log() {
    return R"(%s)";
}
""" % (version,hglog))

prefix = ARGUMENTS.get('prefix', "/usr/local")

env = Environment()
if ARGUMENTS.get('omp',0) or os.environ.get('omp',0):
    env["CXX"]="g++ --std=c++11 -Wno-unused-result -fopenmp"
else:
    env["CXX"]="g++ --std=c++11 -Wno-unused-result"

if int(ARGUMENTS.get('debug',0)):
    env.Append(CXXFLAGS="-g -fno-inline".split())
    env.Append(CCFLAGS="-g".split())
    env.Append(LINKFLAGS="-g".split())
else:
    env.Append(CXXFLAGS="-g -O3 -finline".split())
    env.Append(CCFLAGS="-g".split())

eigeninc = ARGUMENTS.get("eigeninc","???")
if not os.path.isdir(eigeninc): eigeninc = "/usr/local/include/eigen3"
if not os.path.isdir(eigeninc): eigeninc = "/usr/include/eigen3"
if not os.path.isdir(eigeninc): die("cannot find eigen include files")

hdf5inc = ARGUMENTS.get("hdf5inc","???")
if not os.path.isdir(hdf5inc): hdf5inc = "/usr/local/include/hdf5/serial"
if not os.path.isdir(hdf5inc): hdf5inc = "/usr/local/include/hdf5"
if not os.path.isdir(hdf5inc): hdf5inc = "/usr/include/hdf5/serial"
if not os.path.isdir(hdf5inc): hdf5inc = "/usr/include/hdf5"
if not os.path.isdir(hdf5inc): die("cannot find hdf5 include files")

env.Append(CPPPATH=[eigeninc,hdf5inc])
env.Append(LIBS=["hdf5_cpp","zmqpp","zmq","png"])

zmqtest = """
#include <zmqpp/zmqpp.hpp>
zmqpp::message message;
void test() { message.add_raw(0, 0); }
"""
def ZmqTest(context):
    print "Checking for message::add_raw in zmqpp...",
    result = context.TryLink(zmqtest, ".cc")
    context.Result(result)
    return result

custom_tests = dict( ZmqTest = ZmqTest )

conf = Configure(env, custom_tests = custom_tests)

conf.CheckLib("hdf5_cpp") or die("no libhdf5_cpp")
conf.CheckLib("zmqpp") or die("no libzmqpp")
conf.CheckLib("zmq") or die("no libzmq")
conf.CheckLib("png") or die("no libpng")

if conf.CheckLib("hdf5_serial"): conf.env.Append(LIBS=["hdf5_serial"])
elif conf.CheckLib("hdf5"): conf.env.Append(LIBS=["hdf5"])
else: die("did not find -lhdf5_serial or -lhdf5")

if not conf.ZmqTest(): conf.env.Append(CPPDEFINES={'add_raw' : 'add'})

env = conf.Finish()

libs = env["LIBS"]

libclstm = env.StaticLibrary("clstm", source = ["clstm.cc", "extras.cc"])

Alias('install-lib', Install(os.path.join(prefix,"lib"), libclstm))
Alias('install-include', Install(os.path.join(prefix,"include"), ["clstm.h"]))
Alias('install',['install-lib', 'install-include'])

env.Program("clstmctc",
            ["clstmctc.cc", "version.h"],
            LIBS=[libclstm]+libs)
env.Program("clstmseq",
            ["clstmseq.cc", "version.h"],
            LIBS=[libclstm]+libs)
env.Program("clstmconv",
            ["clstmconv.cc", "version.h"],
            LIBS=[libclstm]+libs)
env.Program("clstmtext",
            ["clstmtext.cc", "version.h"],
            LIBS=[libclstm]+libs)
env.Program("clstmimg",
            ["clstmimg.cc", "version.h"],
            LIBS=[libclstm]+libs)
env.Program("test-batch",
            ["test-batch.cc", "version.h"],
            LIBS=[libclstm]+libs)


swigenv = env.Clone( SWIGFLAGS=["-python","-c++"], SHLIBPREFIX="")
swigenv.Append(CPPPATH=[distutils.sysconfig.get_python_inc()])
swigenv.SharedLibrary("_clstm.so",
                      ["clstm.i", "clstm.cc", "extras.cc"],
                      LIBS=libs)

destlib = distutils.sysconfig.get_config_var("DESTLIB")
Alias('pyinstall', Install(os.path.join(destlib, "site-packages"), ["_clstm.so", "clstm.py"]))
