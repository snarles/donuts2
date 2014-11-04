# test precision of nnls

import numpy as np
import numpy.random as npr
import scipy.optimize as spo

nn = 100
pp = 10000

x = npr.exponential(1,(nn,pp))
bb = np.zeros((pp,1))
bb[0:20,0] = (10.)**(-np.array(range(20)))



