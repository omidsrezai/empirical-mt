__author__ = 'orezai'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dropout, merge, Dense, Lambda, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling3D, Conv2D
from keras.constraints import NonNeg
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras.engine.topology import Layer

def buildSelectors_slow(chan_sel,n_filters, sig2 = .09):
    for i in range(n_filters):
        if i==0:
            chans = tf.exp(-tf.square(tf.constant(np.arange(0, n_filters, dtype='float32')) - chan_sel[i])/(2*sig2)) 
            chans = tf.expand_dims(chans,axis=1) 

        else:    
            chan = tf.exp(-tf.square(tf.constant(np.arange(0, n_filters, dtype='float32')) - chan_sel[i])/(2*sig2))      
            chans = tf.concat([chans, tf.expand_dims(chan,axis=1)], axis=-1)
    return chans


def buildSelectors(chan_sel,n_filters, sig2 = .09):

    x = tf.constant(np.arange(0, n_filters, dtype='float32'))
    x = tf.expand_dims(x,axis=1) 
    x = tf.tile(x,[1,n_filters])
    return tf.exp(-tf.square(x - K.transpose(chan_sel))/(2*sig2))


def matVecMul(x,selectors):
    return x*selectors


class SmartInput(Layer):

    def __init__(self, output_dim,regularizer=None, constraint=None, **kwargs):
        self.output_dim = output_dim
        self.combine_weights_seed = np.random.randint(2**32-1)#Seed must be between 0 and 2**32 - 1
        self.initializer=initializers.RandomUniform(minval=0, maxval=1, seed=self.combine_weights_seed)#initializer
        self.regularizer=regularizer
        self.constraint=constraint
        super(SmartInput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.combine_weights = self.add_weight('combine_sp',shape=(input_shape[3], self.output_dim),initializer=self.initializer, 
              regularizer=self.regularizer,constraint=self.constraint, trainable=True)
        #print('SUM',K.sum(self.combine_weights, axis=0, keepdims=False))
        super(SmartInput, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        #print('x', x)
        #print('self.kernel', self.kernel)
        return K.dot(x, self.combine_weights)/K.sum(self.combine_weights, axis=0, keepdims=False)
        #print('j', j)
        #return j

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def get_config(self):
        config = {  
            'output_dim':self.output_dim,
            'kernel_size': self.kernel_size,    
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            #'kernel_initializer': initializers.serialize(self.kernel_initializer),
            #'bias_initializer': initializers.serialize(self.bias_initializer),
            #'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'kernel_constraint': constraints.serialize(self.kernel_constraint),
            #'bias_constraint': constraints.serialize(self.bias_constraint),
            'combine_weights_seed': self.combine_weights_seed
        }
        return confi

class SmartConv2D(Layer):
    def __init__(self, filters, kernel_size, padding="SAME", dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.channel_selector_seed = np.random.randint(2**32-1)#Seed must be between 0 and 2**32 - 1
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer     
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint
        self.data_format = 'channels_last'
        #print('output_dim', output_dim)

        super(SmartConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #print('input_shape[3]', input_shape[3])
        #print('self.output_dim)', self.output_dim)
        self.kernel = self.add_weight(name='kernel_smart',shape=self.kernel_size + (input_shape[3], self.filters),
                                  initializer=self.kernel_initializer,
                                  constraint=self.kernel_constraint,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)


        self.channel_selector = self.add_weight(shape=(self.filters,),
                                    initializer=initializers.RandomUniform(minval=0, maxval=self.filters, seed=self.channel_selector_seed),
                                    name='selector',
                                    regularizer=None,
                                    constraint=NonNeg())

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        super(SmartConv2D, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        #print('x', x)
        #print('self.kernel', self.kernel)
        #outputs = tf.nn.convolution(input=x, filter=kernel, dilation_rate=(self.dilation_rate,), strides=(self.strides,), padding=self.padding, data_format='NHWC')

        selectors = buildSelectors(self.channel_selector,self.filters, sig2 = .09)
        final_kernel = matVecMul(self.kernel,selectors)

        outputs = tf.nn.convolution(x, final_kernel, padding=self.padding, data_format='NHWC')

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:            
            return self.activation(outputs)
        return outputs


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = {  
        'filters': self.filters,
        'kernel_size': self.kernel_size,    
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        #'kernel_initializer': initializers.serialize(self.kernel_initializer),
        #'bias_initializer': initializers.serialize(self.bias_initializer),
        #'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
        #'kernel_constraint': constraints.serialize(self.kernel_constraint),
        #'bias_constraint': constraints.serialize(self.bias_constraint),
        'channel_selector_seed': self.channel_selector_seed
        }
        return config

class AddBiasNonlinear(Layer):
    def __init__(self, filters, activation=None, use_bias=True,bias_initializer='zeros', bias_regularizer= None, bias_constraint=None, **kwargs):
        self.filters = filters
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer=bias_initializer
        self.bias_regularizer=bias_regularizer     
        self.bias_constraint=bias_constraint
        self.data_format = 'channels_last'
        super(AddBiasNonlinear, self).__init__(**kwargs)

    def build(self, input_shape):      
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        super(AddBiasNonlinear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.use_bias:
            outputs = K.bias_add(x, self.bias, data_format=self.data_format)

        if self.activation is not None:            
            return self.activation(outputs)
        return outputs


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = {  
        'filters': self.filters,
        'data_format': self.data_format,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        #'bias_initializer': initializers.serialize(self.bias_initializer),
        #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        #'bias_constraint': constraints.serialize(self.bias_constraint),

        }
        return config


