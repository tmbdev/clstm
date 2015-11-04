FROM ubuntu:15.04
MAINTAINER Tom <tmbdev@gmail.com>
ENV DEBIAN_FRONTEND noninteractive

RUN echo hi
RUN apt-get -qqy update
RUN apt-get -qqy install mercurial
RUN apt-get -qqy install build-essential 
RUN apt-get -qqy install g++ gdb swig2.0 scons
RUN apt-get -qqy install git
RUN apt-get -qqy install wget
RUN cd /tmp && wget -nd http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
RUN cd /tmp && dpkg -i cuda-repo-ubuntu1504*.deb
RUN apt-get -qqy update
RUN apt-get -qqy install cuda-7-5
RUN apt-get -qqy install cuda-command-line-tools-7-5
RUN apt-get -qqy install cuda-core-7-5
RUN apt-get -qqy install cuda-cublas-7-5
RUN apt-get -qqy install cuda-cublas-dev-7-5
RUN apt-get -qqy install cuda-cudart-7-5
RUN apt-get -qqy install cuda-cudart-dev-7-5
RUN apt-get -qqy install cuda-cufft-7-5
RUN apt-get -qqy install cuda-cufft-dev-7-5
RUN apt-get -qqy install cuda-curand-7-5
RUN apt-get -qqy install cuda-curand-dev-7-5
RUN apt-get -qqy install cuda-cusolver-7-5
RUN apt-get -qqy install cuda-cusolver-dev-7-5
RUN apt-get -qqy install cuda-cusparse-7-5
RUN apt-get -qqy install cuda-cusparse-dev-7-5
RUN apt-get -qqy install cuda-minimal-build-7-5
RUN apt-get -qqy install cuda-misc-headers-7-5
RUN apt-get -qqy install cuda-npp-7-5
RUN apt-get -qqy install cuda-npp-dev-7-5
RUN apt-get -qqy install cuda-nvrtc-7-5
RUN apt-get -qqy install cuda-nvrtc-dev-7-5
RUN apt-get -qqy install cuda-runtime-7-5
RUN apt-get -qqy install cuda-toolkit-7-5
RUN apt-get -qqy install cuda-visual-tools-7-5
RUN apt-get -qqy install cuda-samples-7-5

RUN apt-get -qqy install apt-utils
RUN apt-get -qqy install protobuf-compiler libprotobuf-dev
RUN apt-get -qqy install libpng12-dev
RUN cd /usr/local/include && hg clone http://bitbucket.org/eigen/eigen eigen3
RUN apt-get -qqy install strace

RUN apt-get clean && rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/*

VOLUME /work
WORKDIR /work
