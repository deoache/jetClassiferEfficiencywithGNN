#!/bin/bash

export XRD_NETWORKSTACK=IPv4
export X509_USER_PROXY=$HOME/tmp/x509up
cd MAINDIRECTORY

source activate gnn_efficiency
python -V
echo $USER
COMMANDS

if [ "$EOSDIR" != "None" ]; then
    xrdcp -r condor/out/ EOSDIR
fi
