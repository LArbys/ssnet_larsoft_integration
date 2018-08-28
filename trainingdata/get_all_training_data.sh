#!/bin/bash

user=$1
rsync -av --progress ${user}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/larbys/dllee_ssnet_trainingdata/train*.root .
