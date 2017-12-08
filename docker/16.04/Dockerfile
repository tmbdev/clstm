FROM ubuntu:16.04
MAINTAINER Konstantin Baierer <konstantin.baierer@gmail.com>
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update \
    && apt-get install -y \
        git \
        scons \
        g++ \
        libprotobuf-dev \
        libprotobuf9v5 \
        protobuf-compiler \
        libpng12-dev \
        wget \
    && git clone --depth 1 --single-branch --branch 3.3-rc1 \
        "https://github.com/RLovelett/eigen" /usr/local/include/eigen3 \
    && git clone --depth 1 "https://github.com/tmbdev/clstm"

WORKDIR /clstm
RUN scons && scons install && apt-get remove -y g++ scons git
ENV PATH "/clstm:${PATH}"
