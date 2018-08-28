import os,sys
from ROOT import larcv

def load_ssnet_dataset_larcv1( trainingfiles, testingfiles ):
    # define configurations
    if type(trainingfiles) is not list or type(testingfiles) is not list:
        raise ValueError("trainingfiles or testingfiles arguments must be a list of file paths")
    for s in trainingfiles:
        if not os.path.exists(s):
            raise ValueError("File %s does not exist"%(s))
    for s in testingfiles:
        if not os.path.exists(s):
            raise ValueError("File %s does not exist"%(s))
    
    traincfg = """ThreadDatumFillerTrain: {
  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    false
  InputFiles:   {}
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }
    
  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: false
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    validcfg = """ThreadDatumFillerValid: {
  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    false
  InputFiles:   {}
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }
    
  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: false
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    traincfg = traincfg.format( trainingfiles )
    validcfg = validcfg.format( testingfiles )
    with open("ssnet_segfiller_train.cfg",'w') as ftrain:
        print >> ftrain,traincfg
    with open("ssnet_segfiller_valid.cfg",'w') as fvalid:
        print >> fvalid,validcfg
    
    iotrain = LArCV1Dataset("ThreadDatumFillerTrain","ssnet_segfiller_train.cfg" )
    iovalid = LArCV1Dataset("ThreadDatumFillerValid","ssnet_segfiller_valid.cfg" )
    iotrain.init()
    iovalid.init()

    return iotrain,iovalid


if __name__=="__main__":

    trainingfiles = ["val.root"]
    testingfiles  = ["val.root"]
    iotrain, iovalid = load_ssnet_dataset_larcv1( trainingfiles, testingfiles )
    train_nentries = iotrain.io.get_n_entries()
    valid_nentries = iovalid.io.get_n_entries()
    print "Training number of events: ",train_nentries
    print "Validation number of events: ",valid_nentries

    data = iotrain.getbatch(1)
    print data
    print __dict__(data)

    iotrain.stop()
    iovalid.stop()
    
    
