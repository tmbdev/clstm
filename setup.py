import os

from distutils.core import setup, Extension
from Cython.Build import cythonize


def ensure_protobuf():
    exists = os.path.exists("clstm.pb.cc")
    stale = os.path.getctime("clstm.pb.cc") < os.path.getctime("clstm.proto")
    if not exists or stale:
        print "Generating proto file"
        os.system("protoc clstm.proto --cpp_out=.")

ext = Extension(
    "pyclstm",
    sources=['pyclstm.pyx', 'clstm.cc', 'clstm_prefab.cc', 'extras.cc',
             'batches.cc', 'ctc.cc', 'clstm_proto.cc', 'clstm.pb.cc',
             'clstm_compute.cc', 'tensor.cc'],
    include_dirs=['/usr/include/eigen3', '/usr/local/include/eigen3',
                  '/usr/local/include', '/usr/include/hdf5/serial',
                  '/usr/include/hdf5'],
    libraries=['protobuf', 'png'],
    language='c++',
    extra_compile_args=['-std=c++11', '-Wno-unused-result', '-g', '-Ofast',
                        '-DNODISPLAY=1', '-DTHROW=throw', '-DNDEBUG',
                        '-DEIGEN_NO_DEBUG', '-finline', '-ffast-math',
                        '-fno-signaling-nans', '-funsafe-math-optimizations',
                        '-ffinite-math-only', '-march=native'])

ensure_protobuf()
setup(
    name='clstm',
    version='0.1',
    author="Thomas Breuel, Johannes Baiter",
    description="CLSTM Python bindings",
    ext_modules=cythonize([ext]))
