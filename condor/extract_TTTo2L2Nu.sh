#!/bin/bash

export XRD_NETWORKSTACK=IPv4
export X509_USER_PROXY=$HOME/tmp/x509up
cd /afs/cern.ch/user/d/daocampo/public/jetClassiferEfficiencywithGNN

source activate gnn_efficiency
python -V
echo $USER
python main.py extract --dataset=TTTo2L2Nu

if [ "$/eos/user/d/daocampo/btv" != "None" ]; then
    xrdcp -r condor/out/ /eos/user/d/daocampo/btv
fi
