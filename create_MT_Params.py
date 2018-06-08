__author__ = 'orezai'

import numpy as np
import cPickle as pickle
import scipy.stats as stats  
import matplotlib.pyplot as plt

import MT

seed=654
np.random.seed(seed)
#np.random.seed(651)
#np.random.seed(162)
#np.random.seed(379)
#np.random.seed(839)

n_mt = 64# Number of MT channels


params = MT.sample_tuning_params(n_mt, pref_log_speed_range=(0,4))
f1 = open('params_MT_%d.pkl'%seed, 'wb')
pickle.dump(params, f1)
f1.close()

