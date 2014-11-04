# test precision of nnls

import numpy as np
import numpy.random as npr
import scipy.optimize as spo

nn = 100
pp = 10000
x = npr.exponential(1,(nn,pp))
b0 = np.zeros((pp,1))
b0[0:20,0] = (10.)**(-np.array(range(20)))
y = np.dot(x,b0)
bb, temp = spo.nnls(x,np.squeeze(y))
np.squeeze(bb)[0:20]/np.squeeze(b0)[0:20]


