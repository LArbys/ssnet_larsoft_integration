import os,sys,time
import ROOT as rt
from larcv import larcv
from larcvdataset.larcv1dataset import LArCV1Dataset

def load_ssnet_dataset_larcv1( trainingfiles, testingfiles, randomize=True ):
    # define configurations
    if type(trainingfiles) is not list or type(testingfiles) is not list:
        raise ValueError("trainingfiles or testingfiles arguments must be a list of file paths")
    for s in trainingfiles:
        if not os.path.exists(s):
            raise ValueError("File %s does not exist"%(s))
    for s in testingfiles:
        if not os.path.exists(s):
            raise ValueError("File %s does not exist"%(s))
    

    productlists =  [(larcv.kProductImage2D,"wire"),    # adc values
                     (larcv.kProductImage2D,"segment"), # labels
                     (larcv.kProductImage2D,"ts_keyspweight")]
    iotrain = LArCV1Dataset(trainingfiles,productlists,randomize=randomize)
    iovalid = LArCV1Dataset(testingfiles,productlists,randomize=False)

    return iotrain,iovalid


if __name__=="__main__":

    trainingfiles = ["../trainingdata/val.root"]
    testingfiles  = ["../trainingdata/val.root"]
    iotrain, iovalid = load_ssnet_dataset_larcv1( trainingfiles, testingfiles )

    train_nentries = iotrain.io.get_n_entries()
    valid_nentries = iovalid.io.get_n_entries()
    print "Training number of events: ",train_nentries
    print "Validation number of events: ",valid_nentries

    tloop = time.time()
    batchsize = 5
    nbatches = 10
    for i in range(0,nbatches):
        data = iotrain.getbatch( batchsize )
        for d,array in data.items():
            print d,array.shape
        print data["_rse_"]
    ttrain = time.time()-tloop


    tloop = time.time()    
    for i in range(0,nbatches):
        data = iovalid.getbatch( batchsize )
        for d,array in data.items():
            print d,array.shape
        print data["_rse_"]
    tloop = time.time()-tloop

    print "For batch size of %d"%(batchsize)
    print "Time for training set: %.2f secs (%.2f secs/batch)"%(ttrain,ttrain/nbatches)    
    print "Time for validation set: %.2f secs (%.2f secs/batch)"%(tloop,tloop/nbatches)    
    
    
