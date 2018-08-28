#!/bin/bash

source setup.sh

if [ $SSNET_INTEGRATION_USESUBMODULE_LARLITE -eq 1 ]
then
    git submodule init
fi

if [ $SSNET_INTEGRATION_USESUBMODULE_LARCV -eq 1 ]
then
    git submodule init
fi

source configure_larcv1.sh

home=$PWD

cd ${LARLITE_BASEDIR}
make

cd ${LARCV_BASEDIR}
make

cd $home
