#!/bin/bash

. path.sh

#加了path.sh 为了能调用kaldi里面的 compute-eer
#datadir, labelpath, savingpath, GPUid, outputname, numexps

python3 main_wav2vec_mean.py

