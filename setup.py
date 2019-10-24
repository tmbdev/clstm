#!/usr/bin/env python

import os, os.path
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
from os.path import getctime

import distutils.sysconfig
config = distutils.sysconfig.get_config_vars()
# OPT=-DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes
# CFLAGS
for k,v in config.items():
  if type(v)==str and "-W" in v:
    print(k,v)
def remove_warnings(opt):
  opt = opt.split()
  opt = [s for s in opt if not s.startswith("-W")]
  return " ".join(opt)
config["OPT"] = remove_warnings(config["OPT"])
config["CFLAGS"] = remove_warnings(config["CFLAGS"])
config["CONFIGURE_CFLAGS"] = remove_warnings(config["CONFIGURE_CFLAGS"])
config["LDSHARED"] = remove_warnings(config["LDSHARED"])

# hgversion = os.popen("hg -q id").read().strip()
hgversion = "unknown"

include_dirs = ['/usr/include/eigen3', '/usr/local/include/eigen3', '/usr/local/include', '/usr/include/hdf5/serial'] + get_numpy_include_dirs()
swig_opts = ["-c++"] + ["-I" + d for d in include_dirs]
swiglib = os.popen("swig -swiglib").read()[:-1]

if not os.path.exists("clstm.pb.cc") or \
    getctime("clstm.pb.cc") < getctime("clstm.proto"):
  print("making proto file")
  os.system("protoc clstm.proto --cpp_out=.")

clstm = Extension('_clstm',
        libraries = ['png','protobuf'],
        swig_opts = swig_opts,
        include_dirs = include_dirs,
        extra_compile_args = ['-std=c++11','-Wno-sign-compare',
            '-Dadd_raw=add','-DNODISPLAY=1','-DTHROW=throw',
            '-DHGVERSION="\\"'+hgversion+'\\""'],
        sources=['clstm.i','clstm.cc','clstm_prefab.cc','extras.cc', 'batches.cc',
                 'clstm_compute.cc', 'ctc.cc','clstm_proto.cc','clstm.pb.cc'])

from setuptools import setup, find_packages

setup(
    name = 'clstm',
    version = '0.1',
    author      = "Thomas Breuel",
    description = """clstm library bindings""",
    url='https://github.com/tmbdev/clstm',
    license='Apache License 2.0',
    install_requires=[
        'wheel >= 0.33',
    ],
    ext_modules = [clstm],
    py_modules = ["clstm"]
)
