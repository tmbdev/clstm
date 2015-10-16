#!/bin/bash
set -ea
trap "echo clstmocrtrain FAILED" EXIT
echo misc/textline.bin.png > _ocrtest.txt
ntrain=601 hidden=50 lrate=1e-2 save_name=_ocrtest \
    ./clstmocrtrain _ocrtest.txt
load=_ocrtest-600.clstm ./clstmocr _ocrtest.txt | \
    grep -s 'performance analysis' > /dev/null
trap "echo clstmocrtrain OK" EXIT
