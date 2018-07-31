__author__ = 'orezai'

"""
Keras add-on to calculate MT tuning channels.

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.engine.topology import Layer
import MT

def contr_func(c, A,B):
    return (A*c)/(c+B)

class DirectionTuning(Layer):
    def __init__(self, output_dim, params, **kwargs):
        self.output_dim = output_dim
        self.pref_dir = tf.convert_to_tensor(params['pref_dir'][0:output_dim], dtype=tf.float32)
        self.dir_width= tf.convert_to_tensor(params['dir_width'][0:output_dim], dtype=tf.float32)
        self.a_n = tf.convert_to_tensor(params['a_n'][0:output_dim], dtype=tf.float32)

        #self.max_rate = tf.convert_to_tensor(params['max_rate'], dtype=tf.float32)
        #self.back_rate = tf.convert_to_tensor(params['back_rate'], dtype=tf.float32)
        #self.A = tf.convert_to_tensor(params['A'], dtype=tf.float32)
        #self.B = tf.convert_to_tensor(params['B'], dtype=tf.float32)

        super(DirectionTuning, self).__init__(**kwargs)
        

    def build(self, input_shape):
        super(DirectionTuning, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):        
        direction = tf.expand_dims(x, -1)
        dir_scale = tf.exp((tf.cos(direction-self.pref_dir)-1)/self.dir_width)# + self.a_n*tf.exp((nptfcos(direction-self.pref_dir-np.pi)-1)/self.dir_width)
        return dir_scale

    def compute_output_shape(self, input_shape):
        out_sh = (input_shape[0], input_shape[1], input_shape[2], self.output_dim)
        return out_sh

    def get_config(self):
        config = {'output_dim':self.output_dim}
        return config


class SpeedTuning(Layer):
    def __init__(self, output_dim, params, unit_conv,**kwargs):
        self.unit_conv = unit_conv 
        self.output_dim = output_dim        
        self.pref_sp = tf.convert_to_tensor(params['pref_sp'][0:output_dim], dtype=tf.float32)
        self.sp_width = tf.convert_to_tensor(params['sp_width'][0:output_dim], dtype=tf.float32 )
        self.sp_offset = tf.convert_to_tensor(params['sp_offset'][0:output_dim], dtype=tf.float32)

        self.Ap = tf.convert_to_tensor(params['Ap'][0:output_dim], dtype=tf.float32)
        self.Bp = tf.convert_to_tensor(params['Bp'][0:output_dim], dtype=tf.float32)
        self.Ag = tf.convert_to_tensor(params['Ag'][0:output_dim], dtype=tf.float32)
        self.Bg = tf.convert_to_tensor(params['Bg'][0:output_dim], dtype=tf.float32)

        #self.max_rate = tf.convert_to_tensor(params['max_rate'], dtype=tf.float32)
        #self.back_rate = tf.convert_to_tensor(params['back_rate'], dtype=tf.float32)
        #self.A = tf.convert_to_tensor(params['A'], dtype=tf.float32)
        #self.B = tf.convert_to_tensor(params['B'], dtype=tf.float32)

        super(SpeedTuning, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(SpeedTuning, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        ###################################################################################################
        '''
        speed_deg_per_second = tf.expand_dims(x[0], -1) * self.unit_conv
        contrast = tf.expand_dims(x[1], -1)
        q = (speed_deg_per_second+self.sp_offset)/(contr_func(contrast, self.Ap, self.Bp) +self.sp_offset)
        speed_scale = tf.exp(-((tf.log(q)**2)/(2*self.sp_width**2)))
        speed_scale = contr_func(contrast, self.Ag, self.Bg)*speed_scale
        '''
        ####################################################################################

        speed_input = x
        speed_deg_per_second = tf.expand_dims(speed_input, -1) * self.unit_conv
        #contrast = tf.expand_dims(x[1], -1)
        q = (speed_deg_per_second + self.sp_offset) / (self.pref_sp + self.sp_offset)
        speed_scale = tf.exp(-((tf.log(q)**2)/(2*self.sp_width**2)))

        return speed_scale    
    
    def compute_output_shape(self, input_shape):
        speed_input = input_shape
        out_sh = (speed_input[0], speed_input[1], speed_input[2], self.output_dim)
        return out_sh

    def get_config(self):
        config = {'output_dim':self.output_dim}
        return config

class DisparityTuning(Layer):
    def __init__(self, output_dim, params, unit_conv,**kwargs):
        self.unit_conv = unit_conv 
        self.output_dim = output_dim
        self.pref_disp = tf.convert_to_tensor(params['pref_disp'], dtype=tf.float32)
        self.disp_width = tf.convert_to_tensor(params['disp_width'], dtype=tf.float32) 
        self.disp_fq = tf.convert_to_tensor(params['disp_fq'], dtype=tf.float32 )
        self.disp_phase = tf.convert_to_tensor(params['disp_phase'], dtype=tf.float32)


        #self.max_rate = tf.convert_to_tensor(params['max_rate'], dtype=tf.float32)
        #self.back_rate = tf.convert_to_tensor(params['back_rate'], dtype=tf.float32)
        #self.A = tf.convert_to_tensor(params['A'], dtype=tf.float32)
        #self.B = tf.convert_to_tensor(params['B'], dtype=tf.float32)

        super(DisparityTuning, self).__init__(**kwargs)
        

    def build(self, input_shape):
        super(DisparityTuning, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):        
        disparity = tf.expand_dims(x, -1) * self.unit_conv    
        disp_scale = tf.exp(-(disparity-self.pref_disp)**2/(2*self.disp_width**2))*tf.cos(2*np.pi*self.disp_fq*(disparity-self.pref_disp)+self.disp_phase)
                        
        return  disp_scale
        
    def compute_output_shape(self, input_shape):
        out_sh = (input_shape[0], input_shape[1], input_shape[2], self.output_dim)
        return out_sh

    def get_config(self):
        config = {'output_dim':self.output_dim}
        return config


if __name__ == '__main__':   

    np.random.seed(1330)
    n_channel = 256
    n_seq = 1
    n_tent = 10
    w_ = 76#100
    params = MT.sample_tuning_params(n_channel, pref_log_speed_range=(0,4))

    flow_dir = np.random.randint(0,250,(n_seq,w_,w_))
    flow_speed = np.random.randint(0,250,(n_seq,w_,w_))
    contrast = np.random.randint(0,250,(n_seq,w_,w_))
    disparity = np.random.randint(0,250,(n_seq,w_,w_))

    DirectionTuning(n_channel,params)(disparity)
    #direction_tuning(n_channel,params)([flow_dir, flow_speed, contrast, disparity])

    print 'test'






















