__author__ = 'orezai'


import numpy as np
import tensorflow as tf
from keras.layers import Input, merge, Dense, Lambda, Flatten, Reshape, Concatenate, TimeDistributed, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


momentum=.99
n_classes = 27
n_timesteps = 3#12
w_frames = 76 
n_mt = 64

mt_0 = Input(shape=(n_timesteps, w_frames, w_frames, n_mt))


#MT
n_mt2 = n_mt
w_mt = 15
p_mt = 6

mt_1 = TimeDistributed(Conv2D(n_mt2, (w_mt, w_mt), activation='relu', padding='same'))(BatchNormalization(momentum=momentum)(mt_0))
mt_2 = TimeDistributed(Conv2D(n_mt2, (w_mt, w_mt), activation='relu', padding='same'))(BatchNormalization(momentum=momentum)(mt_1))
#print('mt_2.shape: ' + str(mt_2.get_shape))

mt_final = TimeDistributed(MaxPooling2D(pool_size=(p_mt, p_mt)))(BatchNormalization(momentum=momentum)(mt_2))


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
n_output = n_classes
#output_1 = Dense(n_output, input_shape=(n_output,), activation='softmax')(BatchNormalization()(dense_1))
output_1 = Dense(n_output, input_shape=(n_output,), activation='softmax')(BatchNormalization(momentum=momentum)(lstm_1))

print('output_1', output_1)
print('output_1.output_shape', output_1.get_shape)



model = Model(inputs=[mt_0], outputs=[output_1])


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])


class RandomSource:

    def __init__(self):
        pass

    """
    A source of random "data" to test that the network runs.
    """
    def get_data(self, n):
        mt_tuning = np.random.randn(n, n_timesteps, w_frames, w_frames, n_mt)
        act_out = np.random.randint(0, high=27, size = n)
        #print('n', n ) 
        #print(act_out.shape) 
        act_out = to_categorical(act_out, num_classes=n_classes)
        return mt_tuning, act_out


minibatch_size = 16#32

training_source = RandomSource()#JesterSource('train')

def generate_training_data():
    while 1:

        mt_tuning, act_out= training_source.get_data(minibatch_size)
        yield ([mt_tuning], act_out)


validation_source = RandomSource()

def generate_validation_data(validation_steps):
    while 1:    
 
        mt_tuning, act_out = validation_source.get_data(minibatch_size)
        yield ([mt_tuning], act_out)


training_steps = 1000#118562//minibatch_size
validation_steps = 100#14787//minibatch_size

checkpointer = ModelCheckpoint('recog_weights.{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
early_st = EarlyStopping(monitor='val_loss', min_delta=10e-6, patience=10, verbose=0, mode='auto')

h = model.fit_generator(generate_training_data(), epochs=30, callbacks=[checkpointer, early_st],
    validation_data=generate_validation_data(validation_steps), steps_per_epoch=training_steps , validation_steps=validation_steps, verbose=1)


