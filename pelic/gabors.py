import numpy as np

def makeGabors(size, num_of_fq = 4, num_of_ori = 4, deg2pix=10/9., center=None):
    """Building the Gabor filter bank. It has been tested only for 4x4=16 filters."""

    n_filters = num_of_fq*num_of_ori
    filters = np.zeros((size,size,n_filters*2))#.astype(np.complex64)
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
            x0 = y0 = (size-1) / 2 # (size-1) instead of size because if size is even the values won't be symmetric: size=4 -> [-2. -1.  0.  1.] 
    else:
            x0 = center[0] 
            y0 = center[1] 
            
    x = x - x0 
    y = y - y0
    
    R = 2.6390
    F0 = 0.03589/R  #0.03589*R**3; division by R means that the fourth frquency (F4) is 0.25 instead of the third one (F3=0.25; see Tutorial on Gabor Filters, p17).
    

    #print F0*R**np.array([0,1,2,3])
    
    for i in range(num_of_fq):

        Fs = F0*R**i
        F = Fs / (1*deg2pix)

        a = 0.9589 *F;
        b = 1.1866 *F;

        #a = 0.5589;
        #b = 0.69;
        for j in range(num_of_ori):
            thetav = (j*np.pi)/num_of_ori
    
            xprime = x*np.cos(thetav)-y*np.sin(thetav)
            yprime = x*np.sin(thetav)+y*np.cos(thetav)
    
            """#filters[:,:,i+4*j,1]  = (a*b)*np.exp(-np.pi*((a**2)*(xprime**2) + (b**2)*(yprime**2)))*np.exp(1j*2*np.pi*F*(x*np.cos(thetav)+y*np.sin(thetav)))
            This won't work because tensor flow's conv2D does not handle complex functions. So had to separate the real and img in cos and sin parts
            """

            filters[:,:,i+num_of_fq*j]  = (a*b)*np.exp(-np.pi*((a**2)*(xprime**2) + (b**2)*(yprime**2)))*np.cos(2*np.pi*F*(x*np.cos(thetav)+y*np.sin(thetav)))
            filters[:,:,i+num_of_fq*j+n_filters]  = (a*b)*np.exp(-np.pi*((a**2)*(xprime**2) + (b**2)*(yprime**2)))*np.sin(2*np.pi*F*(x*np.cos(thetav)+y*np.sin(thetav)))

    return filters

def makeGaussian(size, sigma=10, center=None):
    """ Make a square gaussian kernel. """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = (size-1) / 2 # (size-1) instead of size because if size is even the values won't be symmetric: size=4 -> [-2. -1.  0.  1.] 
    else:
        x0 = center[0]
        y0 = center[1]   
    h = 1*np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2)) #(1/((2*np.pi)**.5*sigma)) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    return h / np.sum(h.flatten())
