#!/bin/bash

git submodule init

source configure_larcv1.sh

cd ../larlite
make

cd ../larcv1
make

cd ..
