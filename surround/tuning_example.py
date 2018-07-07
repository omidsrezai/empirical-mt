__author__ = 'orezai'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda, Input, TimeDistributed

from keras.models import Model

def lambda0(x):
    input_shape = x.get_shape().as_list()
    return Lambda(lambda x: K.expand_dims(x[:,:,:,0],-1), output_shape= input_shape)(x)

def lambda1(x):
    input_shape = x.get_shape().as_list()
    return Lambda(lambda x: K.expand_dims(x[:,:,:,1],-1), output_shape= input_shape)(x)

def contr_func(c, A,B):
    return (A*c)/(c+B)

class DirectionTuning(Layer):
    def __init__(self, output_dim, params, **kwargs):
        self.output_dim = output_dim
        self.pref_dir = tf.convert_to_tensor(params['pref_dir'], dtype=tf.float32)
        self.dir_width= tf.convert_to_tensor(params['dir_width'], dtype=tf.float32)
        self.a_n = tf.convert_to_tensor(params['a_n'], dtype=tf.float32)        

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
        return conf


class SpeedTuning(Layer):
    def __init__(self, output_dim, params, unit_conv,**kwargs):
        self.unit_conv = unit_conv 
        self.output_dim = output_dim        
        self.pref_sp = tf.convert_to_tensor(params['pref_sp'], dtype=tf.float32)
        self.sp_width = tf.convert_to_tensor(params['sp_width'], dtype=tf.float32 )
        self.sp_offset = tf.convert_to_tensor(params['sp_offset'], dtype=tf.float32) 

        self.Ap = tf.convert_to_tensor(params['Ap'], dtype=tf.float32)
        self.Bp = tf.convert_to_tensor(params['Bp'], dtype=tf.float32)
        self.Ag = tf.convert_to_tensor(params['Ag'], dtype=tf.float32)
        self.Bg = tf.convert_to_tensor(params['Bg'], dtype=tf.float32)

        #self.max_rate = tf.convert_to_tensor(params['max_rate'], dtype=tf.float32)
        #self.back_rate = tf.convert_to_tensor(params['back_rate'], dtype=tf.float32)
        #self.A = tf.convert_to_tensor(params['A'], dtype=tf.float32)
        #self.B = tf.convert_to_tensor(params['B'], dtype=tf.float32)

        super(SpeedTuning, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(SpeedTuning, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):        
        x0 = lambda0(x)

        speed_deg_per_second = x0 *self.unit_conv #tf.expand_dims(x, -1) *self.unit_conv
        #contrast = lambda1(x)
        #q = (speed_deg_per_second+self.sp_offset)/(contr_func(contrast, self.Ap, self.Bp) +self.sp_offset)
        q = (speed_deg_per_second+self.sp_offset)/(self.pref_sp  +self.sp_offset)
        speed_scale = tf.exp(-((tf.log(q)**2)/(2*self.sp_width**2)))
        #speed_scale = contr_func(contrast, self.Ag, self.Bg)*speed_scale
        return speed_scale    
    
    def compute_output_shape(self, input_shape):
        out_sh = (input_shape[0], input_shape[1], input_shape[2], self.output_dim)
        print('out_sh', out_sh)
        return out_sh

    def get_config(self):
        config = {'output_dim':self.output_dim}
        return conf

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
        return conf


if __name__ == '__main__':   

    np.random.seed(1330)
    n_channel = 64
    n_timesteps = 12
    n_seq = 1
    n_tent = 10
    w_ = 76#100
    n_classes = 27

    dataset_length = 118562
    unit_conv = 7.2
    target_file = 'jester-v1-train.csv'
    reader = csv.reader(open('jester-v1-labels.csv',"rb"), delimiter=",")
    labels_dic  = [row for idx, row in enumerate(reader)]

    params = MT3.sample_tuning_params(n_channel, pref_log_speed_range=(0,4))

    flow_dir = np.random.randint(0,250,(n_seq,w_,w_))
    flow_speed = np.random.randint(0,250,(n_seq,w_,w_))
    contrast = np.random.randint(0,250,(n_seq,w_,w_))
    disparity = np.random.randint(0,250,(n_seq,w_,w_))

    move_indexes = np.random.randint(0,dataset_length,n_seq)
    directions, speeds, sp_tent, contrasts, labels_inds = get_labels_fields(move_indexes, target_file, labels_dic, unit_conv, n_tent=n_tent, seq_length=n_timesteps, w_size=w_, h_size=w_, n_classes=n_classes)
    speeds_conts = np.stack([speeds,contrasts], axis=-1)


    flow_speed_cont_tens = Input(shape=(n_timesteps, w_, w_, 2))
    speed_gauss = TimeDistributed(SpeedTuning(n_channel,params,unit_conv))(flow_speed_cont_tens)


##################################################################################

    model = Model(inputs=[flow_speed_cont_tens], outputs=[speed_gauss])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])
 
    qq = model.predict([speeds_conts], batch_size=n_seq, verbose=0)

    for i in range(n_channel):
        plt.subplot(8,8,i+1)
        plt.plot(qq[:,:,:,:,i].flatten(),'o')

    plt.show()
    print('qq', qq.shape)
    rr
#################################################################################

    SpeedTuning(n_channel, params, unit_conv)(speeds_conts)
    DisparityTuning(n_channel,params)(disparity)
    #direction_tuning(n_channel,params)([flow_dir, flow_speed, contrast, disparity])

    print 'test'






















