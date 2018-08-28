# ssnet_larsoft_integration

Scripts and tools for SSNet integration into `uboonecode`.

### Contents

* trainingdata: scripts to pulll down the ssnet training data used for the ssnet-paper
* keras_ssnet: reimplementation of the paper-ssnet network using keras high-level interface to Tensorflow.
  Contains training scripts for training eventual MCC9 network.

### Submodules

We include our dependencies as submodules

* larlite: contains UB constants (for geometry, electronics, etc.)
* larcv1: file IO for images and meta-data
* larcvdataset: python interfaces to larcv1 data. used by tensorflow/pytorch scripts

### Setup

You can setup the repository using `first_setup_larcv1.sh`.

*NOTE* this setup will clone the submodule dependencies, namely larcv1 and larlite.
You might have these somewhere else and what to use those instead.

To do that, set the flags in `setup.sh` to

    export SSNET_INTEGRATION_USESUBMODULE_LARLITE=0
    export SSNET_INTEGRATION_USESUBMODULE_LARCV1=0

Then this code will rely on your other larcv1 and larlite repo. locations.
    


