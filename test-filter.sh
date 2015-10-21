#!/bin/sh
set -ae
rm -f _filter-*.clstm
trap "echo clstmfilter FAILED" EXIT
echo 'hello	hello' > _filter.txt
hidden=20 ntrain=1001 neps=0 report_every=200 save_every=1000 lrate=1e-2 save_name=_filter \
    ./clstmfiltertrain _filter.txt
load=_filter-1000.clstm ./clstmfilter _filter.txt |
    grep -s hello
rm -f _filter-*.clstm _filter.txt
trap "echo clstmfilter OK" EXIT
