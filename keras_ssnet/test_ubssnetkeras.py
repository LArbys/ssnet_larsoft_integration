import os,sys
import ROOT as rt
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from ubssnetkeras import UBSSNetKeras

def pixelwise_crossentropy( y_true, y_predict ):
    x = keras.layers.Subtract()([y_true,y_predict])
    print "after loss sub: ",x
    x = keras.backend.square(x)
    x = keras.backend.sum(x,axis=3)
    print "after loss square+sum: ",x
    x = tf.keras.backend.mean(x,axis=[0,1,2])
    print "after mean reduction: ",x
    return x

#tf.enable_eager_execution()

batch_size=1
inputs = keras.Input(shape=(batch_size,1,512,512))
np_inputs  = np.zeros( (1,512,512,1), dtype=np.float32 )
np_targets = np.ones( (1,512,512,3), dtype=np.float32 )

tbcallbacks = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size,
                            write_graph=True, write_grads=False, write_images=False )


model = UBSSNetKeras( num_classes=3, input_channels=1,
                      inplanes=16, final_conv_kernels=16,
                      showsizes=True, use_group_deconv=True )

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=pixelwise_crossentropy,
              metrics=['accuracy'])


#tf.executing_eagerly()

#model.predict( np_inputs, batch_size=1, verbose=1, steps=None, callbacks=[tbcallbacks] )
model.fit( np_inputs, np_targets, batch_size=1, epochs=1, callbacks=[tbcallbacks] )
