__author__ = 'orezai'

import numpy as np

def filter_kernels(kernel_smart, chan_sel, sig2 = .09):
    '''
    Arguments:
    # kernel_smart: shape (15,15,64,64)
    # chan_sel: shape (64,)

    Output: filtered_kernels: (64,15,15)
    '''
    x = np.arange(0, kernel_smart.shape[-1], dtype='float32')
    x = np.expand_dims(x,axis=1) 
    x = np.tile(x,[1,kernel_smart.shape[-1]])
    kernels = kernel_smart*np.exp(-np.square(x - np.transpose(chan_sel))/(2*sig2))
    filtered_kernels = []

    for i in range(kernel_smart.shape[-1]):
        w_w = kernels[:,:,:,i]    
        foo = np.mean(w_w,axis=0)
        foo = abs(np.mean(foo,axis=0))
        indexx = np.argmax(foo)
        w_w = w_w[:,:,indexx] 
        filtered_kernels.append(w_w)

    return np.array(filtered_kernels)

