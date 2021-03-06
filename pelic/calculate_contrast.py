from __future__ import division

__author__ = 'orezai'

import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imread
import scipy.io as scio

from gabors import makeGabors, makeGaussian

class CalculateContrast:

    def __init__(self, filters, smoother):
        self.n_filter = 16 # These two variables should be inputs from user for more general case (i.e. anything other than 4x4=16)
        self.n_fq = 4
        self.sess = tf.Session()
        self.build_graph()
        self.filters = filters
        self.smoother = smoother

    def build_graph(self):
        """Function to calculate contrast in TensorFlow.
        input_dic = {'images':aa, 'filters':gabors, 'smoother':gaussian, 'is_smoothed':boolean}"""
    
        #Define the graph.
        self.image_tens = tf.placeholder(tf.float32, shape=[None,None,None])
        #batch_size = tf.placeholder(tf.int32)
        self.filters_tens = tf.placeholder(tf.float32, shape=[None, None,self.n_filter*2]) # n_filter*2 instead of n_filter because sin&cos pair for each given freq and ori 
        self.smoother_tens = tf.placeholder(tf.float32, shape=[None, None])
        self.is_smoothed_tens = tf.placeholder(tf.bool) #Whether to smooth/blur the contrast with a Gaussian  

        #tai = tf.placeholder(tf.float32, shape=[None,None,None]) # This was used to make sure original matlab script and this python scritp work the same.

        img_mean, img_var = tf.nn.moments(tf.layers.Flatten()(self.image_tens),1)

        image_expanded = tf.expand_dims(self.image_tens, -1)# Necessary for tf.nn.conv2d (in_channels=1)
        filters_expanded = tf.expand_dims(self.filters_tens,2)# Necessary for tf.nn.conv2d (in_channels=1)
        smoother_expanded = tf.expand_dims(tf.expand_dims(self.smoother_tens,-1),-1)# Necessary for tf.nn.conv2d (in_channels=1, out_channels=1)

        bi = tf.nn.conv2d(image_expanded, filters_expanded, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True, data_format=None, name=None)#bi has n_filter*2 channels (cos&sin)

        #bi->ai by getting rid of the phase of (cos&sin) pairs. ai has n_filter channels 
        for i in range(self.n_filter):
            if i==0:    
                ai = tf.expand_dims(tf.sqrt(tf.square(bi[:,:,:,i])+tf.square(bi[:,:,:,i+self.n_filter])),-1)
            else:
                foo = tf.expand_dims(tf.sqrt(tf.square(bi[:,:,:,i])+tf.square(bi[:,:,:,i+self.n_filter])),-1)
                ai = tf.concat([ai,foo],-1)

        # Tile a tensor with all img_mean for l0
        l0 = tf.expand_dims(img_mean,-1)
        l0 = tf.expand_dims(l0,-1)
        l0 = tf.tile(l0,[1,tf.shape(self.image_tens)[1],tf.shape(self.image_tens)[2]])

        li = tf.expand_dims(l0,-1) 

        #ai = 1321*tf.expand_dims(tai,0)    

        ''' The following for loop builds li (l0_i). An example for 100x100 image size and n_ori=n_freq is:
        l0_0 = tf.expand_dims(l0*tf.ones([100,100],dtype=tf.float32),-1)
        l0_1 = l0 + tf.reduce_sum(ai[:,:,:,0:0] ,axis=3),-1)
        l0_2 = l0 + tf.reduce_sum(ai[:,:,:,0:1] ,axis=3),-1)
        l0_3 = l0 + tf.reduce_sum(ai[:,:,:,0:2] ,axis=3),-1)

        l0_4 = tf.expand_dims(l0*tf.ones([100,100],dtype=tf.float32),-1)
        l0_5 = l0 + tf.reduce_sum(ai[:,:,:,4:4] ,axis=3)
        l0_6 = l0 + tf.reduce_sum(ai[:,:,:,4:5] ,axis=3)
        l0_7 = l0 + tf.reduce_sum(ai[:,:,:,4:6] ,axis=3)

        l0_8 = tf.expand_dims(l0*tf.ones([100,100],dtype=tf.float32),-1)
        l0_9 = l0 + tf.reduce_sum(ai[:,:,:,8:8] ,axis=3)
        l0_10 = l0 + tf.reduce_sum(ai[:,:,:,8:9] ,axis=3)
        l0_11 = l0 + tf.reduce_sum(ai[:,:,:,8:10] ,axis=3)

        l0_12 = tf.expand_dims(l0*tf.ones([100,100],dtype=tf.float32),-1)
        l0_13 = l0 + tf.reduce_sum(ai[:,:,:,12:12] ,axis=3)
        l0_14 = l0 + tf.reduce_sum(ai[:,:,:,12:13] ,axis=3)
        l0_15 = l0 + tf.reduce_sum(ai[:,:,:,12:14] ,axis=3)'''

        for i in range(1,self.n_filter):    
            if i%4==0:
                li = tf.concat([li,tf.expand_dims(l0,-1)],-1)

            else:        
                #print range((i//4)*self.n_fq,(i//4)*self.n_fq+(i%4))
                foo = tf.reduce_sum(ai[:,:,:,(i//4)*self.n_fq:(i//4)*self.n_fq+(i%4)] ,axis=3)
                li = tf.concat([li,tf.expand_dims(l0+foo,-1)],-1)

        ci = ai/li 

        # Combining channels of different ori and freq in a weighted average 
        coeff_freq = tf.constant([1., 1., 1., 1.],dtype=tf.float32) #this did't seem compelling:tf.constant([.1, 1, 3, .5],dtype=tf.float32)#(see De Valoiset al., 1982a, p. 551 Fig 6.b).
        coeff_freq_extended = tf.tile(tf.expand_dims(coeff_freq,-1) , [4,1])

        coeff_ori = tf.constant([3, 2, 3, 2],dtype=tf.float32)#In the area mapping the fovea, there are more kernels oriented vertically and horizontally than...
                            #oriented diagonally (about 3 to 2). (see De Valoiset al., 1982b, p. 537).
        coeff_ori_extended = tf.reshape(tf.tile(tf.expand_dims(coeff_ori,-1) , [1,4]),[16,1])

        ci_bio_norm = tf.expand_dims(ci,-1) *coeff_freq_extended*coeff_ori_extended
        ci_bio_norm = tf.squeeze(ci_bio_norm,-1)
        ci_bio_norm = tf.reduce_sum(ci_bio_norm,axis=-1,keep_dims=True)/ (tf.reduce_sum(coeff_freq)*tf.reduce_sum(coeff_ori))

        #Smoothing contrast map if self.is_smoothed_tens=True
        cont_smoothed = tf.cond(tf.equal(self.is_smoothed_tens, tf.constant(True)), lambda: tf.nn.conv2d(ci_bio_norm, smoother_expanded, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True, data_format=None, name=None), lambda:ci_bio_norm)

        #Rescaling such that mean of the cont_map is std of the image to comply with std definition of contrast
        cont_mean, cont_var = tf.nn.moments(tf.layers.Flatten()(cont_smoothed),1)
        gain = (img_var)**.5/cont_mean #tf.constant([1,2,3.])#
        gain = tf.expand_dims(gain,1)
        cont_final =(tf.transpose(cont_smoothed,perm=[1,2,0,3]))*gain
        cont_final = tf.transpose(cont_final,perm=[2,0,1,3])

        #Average the contrast maps within a batch -----> Should not do this for Jester dataset
        #cont_final_mean_batch,garb = tf.nn.moments(cont_final,0)
        #self.cont_final_mean_batch = tf.squeeze(cont_final_mean_batch)
        self.cont_final_mean_batch = tf.squeeze(cont_final)
        

    def calculate_contrast(self, images, is_smoothed=True):
        self.is_smoothed = is_smoothed
        return self.sess.run(self.cont_final_mean_batch, feed_dict={self.image_tens:images, self.filters_tens:self.filters, self.smoother_tens:self.smoother, self.is_smoothed_tens:self.is_smoothed})


if __name__ == '__main__':
    # Launch the graph.
    size_ = 101
    gabors = makeGabors(size_, center=None)
    gaussian = makeGaussian(size_, sigma=5, center=None)

    CC = CalculateContrast(gabors,gaussian)

    for i in range(1*1):
        a = imread('1.bmp', flatten=True)
        #a = imread('2.png', flatten=True)
    
        plt.imshow(a,cmap='gray')
        #plt.show()
        a = np.expand_dims(a,0)
    
        aa = a
        for i in range(5):
            #print 'i',i
            aa = np.concatenate([aa,a],axis=0)
        print aa.shape

        for i in range(1*1):            
            result = CC.calculate_contrast(aa, is_smoothed=False)
            print result.shape
            plt.imshow(result,cmap='gray')            
            print i
            plt.show()



