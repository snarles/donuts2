# toy model for multiple kernel

# Toy model: heaviside function

import donuts.deconv.utils as du
import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.optimize as spo
import scipy.spatial.distance as dist
import donuts.emd as emd

def heavisides(grid,bvecs):
    x = np.hstack([np.logical_and(bvecs > g[0]-g[1],bvecs < g[0]+g[1]).astype(float) for g in grid])
    x = du.normalize_rows(x.T).T
    return x

# create a grid of measurement vectors
bvecs = np.reshape(np.arange(0.,1.,0.01),(-1,1))
grid = du.fullfact([100,100])/100
grid[:,1] = grid[:,1]+.01

# true positions
true_pos = np.vstack([[.1,.05],[.3,.05],[.7,.2]])
true_w = np.reshape([.1,.2,.1],(-1,1))
true_sigma = 0.1

# generate the true signal

x = heavisides(true_pos,bvecs)                                                                                             
mu = np.dot(x,true_w)
y = mu + np.random.normal(0,1,np.shape(mu))
xx = heavisides(grid,bvecs)
yh, beta, est_pos, est_w = du.lasso_est(y,xx,grid,0.)
