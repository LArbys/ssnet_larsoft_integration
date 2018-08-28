#!/bin/bash


home=$PWD

source setup.sh

if [ $SSNET_INTEGRATION_USESUBMODULE_LARLITE -eq 1 ]
then
    cd larlite
else
    cd ${LARLITE_BASEDIR}
fi
source config/setup.sh || exit
cd $home
    

if [ $SSNET_INTEGRATION_USESUBMODULE_LARCV -eq 1 ]
then
    cd larcv1
else
    cd ${LARCV_BASEDIR}
fi
source configure.sh || exit
cd $home

cd larcvdataset
source setenv.sh
cd $home



