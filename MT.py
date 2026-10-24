__author__ = 'osrezai'

"""
Code to create a population of MT neurons.

Usage: params = MT.sample_tuning_params(n_mt, pref_log_speed_range=(0,4))

"""

import numpy as np
import matplotlib.pyplot as plt
# from dots import make_dot_sequence
import cPickle
from sklearn.mixture import GMM
# MT speed and direction tuning model from Nover

def random_in_range(n, range_):
    return range_[0] + (range_[1]-range_[0])*np.random.rand(n)

def sample_tuning_params(n_neurons, rf_centre_range=(-40,40), rf_rad_range=(5,20), pref_log_speed_range=(0,3.5)):
	#Direction	
	pref_dir = 0 + (2*np.pi)*np.random.rand(n_neurons)
	dir_bw =  np.random.gamma(7.32, scale=14.25, size=n_neurons)
	dir_width= (np.cos(np.pi/180*(np.minimum(dir_bw,360)/2))-1)/(-np.log(2))*1 #(np.cos(np.pi/180*(dir_bw/2))-1)/(-np.log(2))
	a_n = np.zeros(n_neurons) #'t location-scale'

	#Speed
	low_sp = np.log(.1)
	Hi_sp = np.log(32.)

	pref_sp = np.exp(low_sp + (Hi_sp-low_sp)*np.random.rand(n_neurons))
	sp_width =  np.random.gamma(4.36, scale=0.28, size=n_neurons)
	sp_offset = np.random.gamma(0.88, scale=0.58, size=n_neurons)

	#Disparity
	mu = -1#-0.20
	sigma = 0.62
	pref_disp = mu + sigma*np.random.standard_t(9.67, size=n_neurons)
	disp_width = 1.4 + np.random.rand(n_neurons) ####################
	disp_fq = np.random.lognormal(mean = 0.46, sigma = 0.82, size = n_neurons)
	
	gmm = GMM(2, n_iter=1)
	gmm.means_ = np.array([[1.32], [-1.32]])
	gmm.covars_ = np.array([[0.63], [0.81]]) ** 2
	gmm.weights_ = np.array([0.4, 0.6])
	disp_phase = np.squeeze(gmm.sample(n_neurons))

	#Contrast (Check find_contrast_dist.py)
	low_Ap = np.log(.2)
	Hi_Ap = np.log(128.)

	Ap = np.exp(low_Ap + (Hi_Ap-low_Ap)*np.random.rand(n_neurons))
	Bp = 5 * np.random.randn(n_neurons) + 50
	
	low_Ag = 0.8
	Hi_Ag = 1.1

	Ag = low_Ag + (Hi_Ag-low_Ag)*np.random.rand(n_neurons)
	Bg = 1 * np.random.randn(n_neurons) + 5
	
	#Nonlinearity
	max_rate = 150 + (100)*np.random.rand(n_neurons)
	back_rate = 2 + (3)*np.random.rand(n_neurons)

	speed_deg_per_second = 0
	direction = 0
	disparity = 0

	A = []
	B = []

	'''
	for i in range(n_neurons):
		q = (speed_deg_per_second+sp_offset[i])/(pref_sp[i] +sp_offset[i])
		speed_scale = np.exp(-((np.log(q)**2)/(2*sp_width[i]**2)))
		dir_scale = np.exp((np.cos(direction-pref_dir[i])-1)/dir_width[i]) + a_n[i]*np.exp((np.cos(direction-pref_dir[i]-np.pi)-1)/dir_width[i])
		disparity_scale = np.exp(-(disparity-pref_disp[i])**2/(2*disp_width[i]**2))*np.cos(2*np.pi*disp_fq[i]*(disparity-pref_disp[i])+disp_phase[i])
        	r = speed_scale * dir_scale * disparity_scale
		a = (max_rate[i] - back_rate[i])/(1-r)
		A.append(a) 
		B.append(max_rate[i] - a) 
	A = np.squeeze(np.array(A))
	B = np.squeeze(np.array(B))
	'''
	params = {'pref_dir': pref_dir, 'dir_width':dir_width, 'a_n':a_n, 'pref_sp':pref_sp, 'sp_width':sp_width, 'sp_offset':sp_offset, \
 'pref_disp':pref_disp, 'disp_width':disp_width, 'disp_fq':disp_fq, 'disp_phase':disp_phase, 'Ap':Ap, 'Bp':Bp, 'Ag':Ag, 'Bg':Bg, 'max_rate': max_rate, 'back_rate':back_rate, 'A':A, 'B':B,
			  'n_chann': n_neurons}
	return params

'''
if __name__ == '__main__':
	n_neurons = 50
	tunings = sample_tuning_params(n_neurons)

	print('dir_width',tunings['dir_width'].shape)
	print('Ag',tunings['Ag'].shape,tunings['Bg'].shape,tunings['Ap'].shape,tunings['Bg'].shape)
	print('tunings[max_rate]',tunings['max_rate'].shape)
	print('disp_phase',tunings['disp_phase'].shape)
'''
# nbins = 30
# plt.subplot(3,2,1), plt.hist(tunings.cx, nbins), plt.title('cx')
# plt.subplot(3,2,2), plt.hist(tunings.cy, nbins), plt.title('cy')
# plt.subplot(3,2,3), plt.hist(tunings.rad, nbins), plt.title('rad')
# plt.subplot(3,2,4), plt.hist(tunings.pref_log_speed, nbins), plt.title('pref log speed')
# plt.subplot(3,2,5), plt.hist(tunings.width_log_speed, nbins), plt.title('width log speed')
# plt.subplot(3,2,6), plt.hist(tunings.width_dir, nbins), plt.title('width dir')
# plt.show()

def sample_patch_params(n, max_log_speed=3.5, centre_range=(-20,20)):
    # result = Patches()
    # result.cx = random_in_range(n, centre_range)
    # result.cy = random_in_range(n, centre_range)
    speed = np.exp(random_in_range(n, (0,max_log_speed)))
    dir_ = random_in_range(n, (0,2*np.pi))
    disp_ = random_in_range(n, (-6,.5))
    # result.rad = random_in_range(n, (5,10)) #degrees visual angle
    return {'speed': speed, 'dir': dir_}

'''
n_patches = 30000
patches = sample_patch_params(n_patches)

# im_size = (100,100)
# pixels_per_degree = 2
# imx = (np.arange(im_size[1]) - im_size[1]/2 + 0.5) / pixels_per_degree
# imy = (np.arange(im_size[0]) - im_size[0]/2 + 0.5) / pixels_per_degree
#
# IMX, IMY = np.meshgrid(imx, imy)

# note: see also scipy.integrate.dblquad

def gaussian(x, mu, sig):
    # return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-(x - mu)**2/sig/2)
    return np.exp(-(x - mu)**2/sig/2)
'''

if __name__ == '__main__':

    n_neurons = 10
    tunings = sample_tuning_params(n_neurons)

    disparity = np.arange(-10,10,.1)

    disp_scale = []
    for j in range(n_neurons):
        disp_scale = (np.exp(-(disparity-tunings['pref_disp'][j])**2/(2*tunings['disp_width'][j]**2))*np.cos(2*np.pi*tunings['disp_fq'][j]*(disparity-tunings['pref_disp'][j])+tunings['disp_phase'][j]))
	plt.hold(True)
        plt.plot(disparity, disp_scale)

    plt.show()
    print np.array(disp_scale).shape
    ff
    n_patches = 30000
    patches = sample_patch_params(n_patches)

    responses = np.zeros((n_patches, n_neurons))
    for i in range(n_patches):
        print(i)
        # mask = (IMX-patches.cx[i])**2 + (IMY-patches.cy[i])**2 <= patches.rad[i]**2
        log_speed = np.log(patches['speed'][i])
        dir_ = patches['dir'][i]
        dir_ = patches['dir'][i]

        for j in range(n_neurons):
            # rf_centre = np.exp( -((IMX-tunings.cx[j])**2 + (IMY-tunings.cy[j])**2) / (2*tunings.rad[j]**2) )
            # print(tunings.cx[j])
            # print(tunings.cy[j])
            # print(tunings.rad[j])
            # plt.imshow(rf_centre)
            # plt.colorbar()
            # plt.show()
            #TODO: scale, surround

            # rf_scale = np.sum(rf_centre * mask) / np.sum(rf_centre)
            speed_scale = gaussian(log_speed, tunings['pref_log_speed'][j], tunings['width_log_speed'][j])
            dir_scale = gaussian(dir_, tunings['pref_dir'][j], tunings['width_dir'][j])
            disp_scale = np.exp(-(disparity-tunings['pref_disp'][j])**2/(2*tunings['disp_width'][j]**2))*np.cos(2*np.pi*tunings['disp_fq'][j]*(disparity-tunings['pref_disp'][j])+tunings['disp_phase'][j])
            # responses[i,j] = rf_scale * speed_scale * dir_scale

            responses[i,j] = speed_scale * dir_scale * disp_scale 
            # responses[i,j] = dir_scale

        speed_pixels = np.exp(log_speed)/30. # should be 60, but I'm worried about quantization
        # make_dot_sequence(dir, speed_pixels, seq=i)

    outfile = open('../data/MT-patches.pkl', 'wb')
    # data = (patches, responses)
    cPickle.dump((patches, responses), outfile)
    outfile.close()

    # plt.hist(responses)
    # plt.show()

    # plt.imshow(responses)
    # plt.colorbar()
    # plt.show()
    # plt.plot(responses[0,:])
    # plt.show()


