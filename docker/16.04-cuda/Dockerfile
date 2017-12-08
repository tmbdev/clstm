FROM ubuntu:16.04
MAINTAINER Konstantin Baierer <konstantin.baierer@gmail.com>
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_URL http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb

WORKDIR /tmp
RUN apt-get -y update \
    && apt-get -y install wget git scons g++ \
        libprotobuf-dev libprotobuf9v5 protobuf-compiler libpng12-dev
# RUN apt-get -y install build-essential gdb strace \
RUN git clone --depth 1 --single-branch --branch 3.3-rc1 \
        "https://github.com/RLovelett/eigen" /usr/local/include/eigen3
RUN wget -nd $CUDA_URL \
    && dpkg -i cuda-repo-ubuntu*.deb \
    && apt-get -y update \
    && apt-get -y install \
        cuda-8-0 \
        cuda-command-line-tools-8-0  \
        cuda-core-8-0  \
        cuda-cublas-8-0  \
        cuda-cublas-dev-8-0  \
        cuda-cudart-8-0  \
        cuda-cudart-dev-8-0  \
        cuda-cufft-8-0  \
        cuda-cufft-dev-8-0  \
        cuda-curand-8-0  \
        cuda-curand-dev-8-0  \
        cuda-cusolver-8-0  \
        cuda-cusolver-dev-8-0  \
        cuda-cusparse-8-0  \
        cuda-cusparse-dev-8-0  \
        cuda-minimal-build-8-0  \
        cuda-misc-headers-8-0  \
        cuda-npp-8-0  \
        cuda-npp-dev-8-0  \
        cuda-nvrtc-8-0  \
        cuda-nvrtc-dev-8-0  \
        cuda-runtime-8-0  \
        cuda-toolkit-8-0  \
        cuda-visual-tools-8-0  \
        cuda-samples-8-0

RUN git clone --depth 1 "https://github.com/tmbdev/clstm"
WORKDIR /tmp/clstm
RUN scons && scons all

VOLUME /work
WORKDIR /work
