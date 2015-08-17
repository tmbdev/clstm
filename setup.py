#!/usr/bin/env python

import os
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

# hgversion = os.popen("hg -q id").read().strip()
hgversion = "unknown"

include_dirs = ['/usr/include/eigen3', '/usr/local/include/eigen3', '/usr/local/include', '/usr/include/hdf5/serial'] + get_numpy_include_dirs()
swig_opts = ["-c++"] + ["-I" + d for d in include_dirs]
swiglib = os.popen("swig -swiglib").read()[:-1]

print "making proto file"
os.system("protoc clstm.proto")

clstm = Extension('_clstm',
        libraries = ['png','protobuf'],
        swig_opts = swig_opts,
        include_dirs = include_dirs,
        extra_compile_args = ['-std=c++11','-Wno-sign-compare',
            '-Dadd_raw=add','-DNODISPLAY=1','-DTHROW=throw',
            '-DHGVERSION="\\"'+hgversion+'\\""'],
        sources=['clstm.i','clstm.cc','clstm_prefab.cc','extras.cc',
                 'ctc.cc','clstm_proto.cc','clstm.pb.cc'])

setup (name = 'clstm',
       version = '0.0',
       author      = "Thomas Breuel",
       description = """clstm library bindings""",
       ext_modules = [clstm],
       py_modules = ["clstm"])
