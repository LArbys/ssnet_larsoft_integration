import os,sys
import tensorflow as tf
import tensorflow.keras as keras

class ResNetLayer(keras.Model):
    def __init__( self, inputchs, outputchs, stride,name="res_net_layer" ):
        super(ResNetLayer,self).__init__(name=name)
        self.outputchs = outputchs
        self.conv1 = keras.layers.Conv2D( filters=inputchs, kernel_size=3, strides=stride, padding="same", use_bias=False  )
        self.bn1   = keras.layers.BatchNormalization( axis=1 )
        self.relu1 = keras.layers.ReLU( )
        self.conv2 = keras.layers.Conv2D( filters=outputchs, kernel_size=3, strides=1, padding="same", use_bias=False  )
        self.bn2   = keras.layers.BatchNormalization( axis=1 )
        self.bypass = None
        self.bnpass = None
        self.byadd  = None
        if inputchs!=outputchs or stride>1:
            self.bypass = keras.layers.Conv2D( filters=outputchs, kernel_size=1, strides=stride, padding="valid", use_bias=False )
            self.bnpass = keras.layers.BatchNormalization( axis=1 )
        self.relu  = keras.layers.ReLU( )

    def call(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.bypass is not None:
            outbp = self.bypass(x)
            outbp = self.bnpass(outbp)
        else:
            outbp = x
        out = keras.layers.add( [out,outbp] )
        out = self.relu(out)
        return out

    def compute_output_shape(self,input_shape):
        """ must provide concrete function for subclassed keras.Model """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.outputchs
        return tf.TensorShape(shape)

class DoubleResNetLayer(keras.Model):
    def __init__( self, inputchs, outputchs, stride,name="double_res_net_layer" ):
        super(DoubleResNetLayer,self).__init__(name=name)
        self.outputchs = outputchs
        self.res1 = ResNetLayer(inputchs,outputchs,stride,name=self.name+"/res_net_layer")
        self.res2 = ResNetLayer(outputchs,outputchs,1,name=self.name+"/res_net_layer")
    def call(self,x):
        out = self.res1(x)
        out = self.res2(out)
        return out
    def compute_output_shape(self,input_shape):
        """ must provide concrete function for subclassed keras.Model """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.outputchs
        return tf.TensorShape(shape)
    

class ConvTransposeLayer(keras.Model):
    def __init__( self, inputchs, outputchs, res_outputchs, group_deconv, stride, name="conv_transpose_layer"):
        super(ConvTransposeLayer,self).__init__(name=name)
        self.outputchs = outputchs
        self.group_deconv = group_deconv
        self.deconvlist = []
        self.splitlist  = []
        if not self.group_deconv:
            convtran = keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(stride,stride), padding='same',use_bias=False,name=self.name+"/deconv")
            self.deconvlist.append(convtran)
            self.concat = None
        else:
            for n in xrange(self.outputchs):
                convtrans = keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(stride,stride), padding='same',use_bias=False,name=self.name+"/deconv")
                self.deconvlist.append(convtrans)
                split    = keras.layers.Lambda( lambda y : y[:,:,:,2*n:2*(n+1)],name=self.name+"/split")       
                self.splitlist.append(split)
            self.concatsplit = keras.layers.Concatenate(axis=3,name=self.name+"/concatsplit")
        self.concat = keras.layers.Concatenate(axis=3,name=self.name+"/concatskip")
        self.dblres = DoubleResNetLayer(res_outputchs+outputchs,res_outputchs,1,name=self.name+"/doubleres")
        
    def call(self,x,x_enc):
        if not self.group_deconv:
            x = self.deconvlist[0].call( x, x_enc_out[n] )
        else:
            # if we use grouped deconvolution, we have to split, convtranspose, concat, resnet
            nsplits = len(self.deconvlist) # nconvtranspose + double res layer
            x_split = []
            for i in xrange(nsplits):
                xtmp = self.splitlist[i](x)
                #print "Shape after split[",i,"]: ",xtmp
                xtmp = self.deconvlist[i](xtmp)
                #print "Shape after deconv2d[",i,"]: ",xtmp
                x_split.append( xtmp )
            x = self.concatsplit( x_split )
            print "Shape after deconv split+concat: ",x
        # now concat with encoder layer
        x = self.concat( [x,x_enc] )
        print "Shape after concat w/ encoder: ",x
        # resnet with concat
        x = self.dblres(x)
        return x
    
    def compute_output_shape(self,input_shape):
        """ must provide concrete function for subclassed keras.Model """
        shape = tf.TensorShape(input_shape).as_list()
        shape[1]  *= 2
        shape[2]  *= 2
        shape[3]  = self.outputchs
        return tf.TensorShape(shape)
    

class UBSSNetKeras(keras.Model):

    def __init__(self, num_classes=3, input_channels=3, inplanes=16, final_conv_kernels=16, showsizes=False, use_group_deconv=True, name="UBSSNetKeras"):
        super(UBSSNetKeras, self).__init__(name=name)
        """ UB SSNet Model (as of 2017/2018 paper) implemented in tensorflow.Keras
        input
        -----
        num_classes: number of classes. default 3 -- background, track, shower
        input_channels: default 3 -- xxx
        inplanes: 16 first encoder channels
        final_conv_kernels: channels in last layer before softmax
        showsizes: dump information about layers when calling call, for debug
        use_group_deconv: use grouped deconvolution (ugh)
        """
        
        self.inplanes = inplanes
        self.num_classes = num_classes
        #self._showsizes = showsizes # print size at each layer
        self._showsizes = True
        self.use_group_deconv = use_group_deconv
        
        # Layers
        self._define_stem_layers( input_channels, inplanes )
        self._define_encoder_layers()
        self._define_decoder_layers()
        self._define_final_layers()

    def _define_stem_layers( self, input_channels, inplanes ):
        """ define stem layers. added to class """
        self.stem_conv1 = keras.layers.Conv2D( filters=self.inplanes, kernel_size=7, strides=1, padding="same", use_bias=True, name="stem/conv" )
        self.stem_bn1   = keras.layers.BatchNormalization( axis=1, name="stem/batchnorm" )
        self.stem_relu1 = keras.layers.ReLU(name="stem/relu")
        self.stem_pool1 = keras.layers.MaxPool2D( pool_size=(2,2), name="stem/pool" )

    def _define_encoder_layers(self):
        self.enc_layers = []
        inchs  = self.inplanes
        outchs = self.inplanes*2
        for l in xrange(5):
            with tf.name_scope("encoding_layer_%d"%(l)) and tf.variable_scope("encoding_layer_%d"%(l)):
                if l==0:
                    enc_layer = self._make_encoding_layer( inchs, outchs, stride=1, name="encoding_layer_%d"%(l) )
                else:
                    enc_layer = self._make_encoding_layer( inchs, outchs, stride=2, name="encoding_layer_%d"%(l) )
                super(UBSSNetKeras,self).__setattr__("enc_layer_%d"%(l),enc_layer) # register attribute to model
                self.enc_layers.append(enc_layer)
            inchs *= 2
            outchs = inchs*2


    def _make_encoding_layer( self, inputchs, outputchs, stride, name  ):
        return DoubleResNetLayer(inputchs,outputchs,stride, name=name )

    def _define_decoder_layers(self):
        self.dec_layers = []
        inchs  = self.inplanes*32
        outchs = self.inplanes*16
        for l in xrange(5):
            dec_layer = self._make_decoding_layer( inchs,  outchs, outchs, 2, self.use_group_deconv, l ) # 512->256
            self.dec_layers.append(dec_layer)
            inchs  = outchs
            outchs = inchs/2
            
        
    def _make_decoding_layer(self, inputchs, outputchs, res_outputchs, stride, group_conv, deconv_level ):    
        with tf.name_scope("deconv_layer_%d"%(deconv_level)):
            deconv = ConvTransposeLayer(inputchs,outputchs,res_outputchs,group_conv,stride,name="deconv_layer_%d"%(deconv_level))
            return deconv

    def _define_final_layers( self ):
        """ define final layers. added to class """
        self.conv10 = keras.layers.Conv2D( filters=self.inplanes, kernel_size=7, strides=1, padding="same", use_bias=True, name="final/conv10" )
        self.bn10   = keras.layers.BatchNormalization( axis=1, name="final/bn10" )
        self.relu10 = keras.layers.ReLU(name="final/relu10")
        self.conv11 = keras.layers.Conv2D( filters=self.num_classes, kernel_size=7, strides=1, padding="same", use_bias=True, name="final/conv11" )
        self.softmax = keras.layers.Softmax(axis=1)

    def call(self,x):

        # first the stem
        x = self.stem_conv1(x)
        x = self.stem_bn1(x)
        x = self.stem_relu1(x)
        x0 = x
        x = self.stem_pool1(x)
        if self._showsizes:
            print "Shape after stem: ",x
            

        # encoders
        x_enc_out = [x0]
        for n in xrange(len(self.enc_layers)):
            enc_out = self.enc_layers[n].call(x)
            x = enc_out
            if self._showsizes:                
                print "Shape after Encoder[",n,"]: ",x
            x_enc_out.append(enc_out)
        x_enc_out.reverse()

        # decoders
        x = x_enc_out[0]
        for n in xrange(len(self.dec_layers)):
            deconv = self.dec_layers[n]
            x = deconv.call(x,x_enc_out[n+1])
            print "Shape after decoder[",n,"] (concat deconv+enc): ",x


        # # final layer
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.conv11(x)
        print "shape after final convlayer: ",x
        x = self.softmax(x)
        print "shape after softmax: ",x

        return x

    def compute_output_shape(self,input_shape):
        """ must provide concrete function for subclassed keras.Model 
        We return a softmax vector at each pixel
        """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

        
        
                    
                    
                
