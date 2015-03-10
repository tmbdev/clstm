import os
import distutils.sysconfig

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
env["CXX"]="g++ --std=c++11 -Wno-unused-result"
if int(ARGUMENTS.get('debug',0)):
    env.Append(CXXFLAGS="-g -fno-inline".split())
    env.Append(CCFLAGS="-g".split())
    env.Append(LINKFLAGS="-g".split())
else:
    env.Append(CXXFLAGS="-g -O3 -finline".split())
    env.Append(CCFLAGS="-g".split())
env.Append(CPPPATH=["/usr/include/eigen3"])
env.Append(LIBS=["hdf5_cpp","hdf5"])
env.Append(LIBS=["png"])
if ARGUMENTS.get('oldzmqpp',0) or os.environ.get('oldzmqpp',0):
    env.Append(CPPDEFINES={'add_raw' : 'add'})

libclstm = env.StaticLibrary("clstm", source = ["clstm.cc", "extras.cc"])

Alias('install-lib', Install(os.path.join(prefix,"lib"), libclstm))
Alias('install-include', Install(os.path.join(prefix,"include"), ["clstm.h"]))
Alias('install',['install-lib', 'install-include'])

# print env.Dump()

env.Program("clstmctc",
            ["clstmctc.cc", "version.h"],
            LIBS=[libclstm,"hdf5_cpp","hdf5","zmqpp","zmq","png"])
env.Program("clstmseq",
            ["clstmseq.cc", "version.h"],
            LIBS=[libclstm,"hdf5_cpp","hdf5","zmqpp","zmq","png"])
env.Program("clstmtext",
            ["clstmtext.cc", "version.h"],
            LIBS=[libclstm,"hdf5_cpp","hdf5","zmqpp","zmq","png", "boost_locale"])
