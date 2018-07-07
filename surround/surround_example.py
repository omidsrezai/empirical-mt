__author__ = 'orezai'

import numpy as np
import tensorflow as tf
import pickle
from keras.layers import Input, merge, Dense, Lambda, Flatten, Reshape, Concatenate, TimeDistributed, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.constraints import NonNeg,NonPos
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.merge import Add, Multiply
from keras import backend as K
from keras.utils import to_categorical
from tuning_example import DirectionTuning, SpeedTuning
from smart_example import SmartInput, SmartConv2D, AddBiasNonlinear
from tent import get_tent_responses

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

momentum=.99
n_timesteps = 5#12
w_frames = 76 
n_mt = 64
n_tent = 10

unit_conv_sp = 4# This is the speed multiplier you choose

#################################################################
params = pickle.load(open('params_recog_sp64_654.pkl','rb')) # You can create this file by running create_params.py

flow_dir_tens = Input(shape=(n_timesteps, w_frames, w_frames))
flow_speed_cont_tens = Input(shape=(n_timesteps, w_frames, w_frames,2)) # when you want to feed this tensor using numpy arrays, stack speed and contrast like this: speeds_conts = np.stack([speeds,contrasts], axis=-1)
speed_tent_tens = Input(shape=(n_timesteps, w_frames, w_frames, n_tent))

dir_ = TimeDistributed(DirectionTuning(n_mt,params))(flow_dir_tens)
print('dir_.shape: ' + str(dir_.get_shape))
speed_gauss = TimeDistributed(SpeedTuning(n_mt,params,unit_conv_sp))(flow_speed_cont_tens)

print('speed_gauss_.shape: ' + str(speed_gauss.get_shape))

#combine input
speed_N =  TimeDistributed(SmartInput(n_mt, regularizer=None, constraint=NonNeg()), name='SmartInput1')(speed_tent_tens)
speed_NS = TimeDistributed(SmartInput(n_mt, regularizer=None, constraint=NonNeg()), name='SmartInput2')(speed_tent_tens)

speed_dir_P = Multiply()([dir_,speed_gauss])
speed_dir_N = Multiply()([dir_,speed_N])

####################################################################

#MT
n_mt2 = n_mt
w_mt = 15
p_mt = 6

mt_1p = TimeDistributed(SmartConv2D(n_mt, (w_mt, w_mt), activation=None,  use_bias=False, padding="SAME", kernel_constraint=NonNeg()), name='exc')(TimeDistributed(BatchNormalization())(speed_dir_P))
mt_1N = TimeDistributed(SmartConv2D(n_mt, (w_mt, w_mt), activation=None,  use_bias=False, padding="SAME", kernel_constraint=NonPos()), name='sup')(TimeDistributed(BatchNormalization())(speed_dir_N))
mt_1NS = TimeDistributed(SmartConv2D(n_mt, (w_mt, w_mt), activation=None, use_bias=False, padding="SAME", kernel_constraint=NonPos()), name='nsup')(TimeDistributed(BatchNormalization())(speed_NS))

mt_1s = Add()([mt_1p, mt_1N,mt_1NS])
#print('mt_1s.shape: ' + str(mt_1s.get_shape))

mt_1 = TimeDistributed(AddBiasNonlinear(n_mt, activation='relu',  use_bias=True))(mt_1s)
mt_2 = TimeDistributed(Conv2D(n_mt2, (w_mt, w_mt), activation='relu', padding='same'))(TimeDistributed(BatchNormalization())(mt_1))

mt_final = TimeDistributed(MaxPooling2D(pool_size=(p_mt, p_mt)))(TimeDistributed(BatchNormalization())(mt_2))

# MST
n_MST = 64
w_MST = 9
p_MST = 3

MST_1 = TimeDistributed(Conv2D(n_MST, (w_MST, w_MST), activation='relu', padding='same'))(BatchNormalization(momentum=momentum)(mt_final))
MST_2 = TimeDistributed(Conv2D(n_MST, (w_MST, w_MST), activation='relu', padding='same'))(BatchNormalization(momentum=momentum)(MST_1))
MST_final = TimeDistributed(MaxPooling2D(pool_size=(p_MST, p_MST)))(BatchNormalization(momentum=momentum)(MST_2))
print('MST_final.shape: ' + str(MST_final.get_shape))

MST_finalb = TimeDistributed(Flatten())(MST_final)
print('MST_finalb.shape: ' + str(MST_finalb.get_shape))

#LSTM

n_lstm = 256#32
#w_lstm = 2
lstm_1 = LSTM(n_lstm, recurrent_dropout=0.5)(BatchNormalization(momentum=momentum)(MST_finalb))
print('lstm_1.shape: ' + str(lstm_1.get_shape))


#output
n_output = 4
#output_1 = Dense(n_output, input_shape=(n_output,), activation='softmax')(BatchNormalization()(dense_1))
output_1 = Dense(n_output, input_shape=(n_output,), activation=None)(BatchNormalization(momentum=momentum)(lstm_1))

print('output_1', output_1)
print('output_1.output_shape', output_1.get_shape)


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = Model(inputs=[flow_dir_tens, flow_speed_cont_tens, speed_tent_tens], outputs=[output_1])

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])
model.summary()

class RandomSource:

    def __init__(self):
        pass

    """
    A source of random "data" to test that the network runs.
    """
    def get_data(self, n):
        flow_dir = np.random.randn(n, n_timesteps, w_frames, w_frames)
        flow_speed = np.random.randn(n, n_timesteps, w_frames, w_frames)
        contrast = np.random.randn(n, n_timesteps, w_frames, w_frames) 
        flow_speed_cont = np.stack([flow_speed,contrast], axis=-1)         
        speed_tent = get_tent_responses(flow_speed,n_tent, unit_conv_sp)
        y_cords = np.random.rand(n,4)
        #print('n', n ) 
        #print(act_out.shape) 
        return flow_dir, flow_speed_cont, speed_tent, y_cords

minibatch_size = 16#32

training_source = RandomSource()#JesterSource('train')

def generate_training_data():
    while 1:

        flow_dir, flow_speed_cont, speed_tent, y_cords= training_source.get_data(minibatch_size)
        yield ([flow_dir, flow_speed_cont, speed_tent], y_cords)


validation_source = RandomSource()

def generate_validation_data(validation_steps):
    while 1:    
 
        flow_dir, flow_speed_cont, speed_tent, y_cords= validation_source.get_data(minibatch_size)
        yield ([flow_dir, flow_speed_cont, speed_tent], y_cords)


training_steps = 1000#118562//minibatch_size
validation_steps = 100#14787//minibatch_size

checkpointer = ModelCheckpoint('recog_weights.{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
early_st = EarlyStopping(monitor='val_loss', min_delta=10e-6, patience=10, verbose=0, mode='auto')

h = model.fit_generator(generate_training_data(), epochs=30, callbacks=[checkpointer, early_st],
    validation_data=generate_validation_data(validation_steps), steps_per_epoch=training_steps , validation_steps=validation_steps, verbose=1)


