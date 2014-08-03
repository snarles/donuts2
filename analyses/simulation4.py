# jackknife estimation for kappa

import numpy as np
import numpy.linalg as nl
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
import matplotlib.pyplot as plt

def get_hatmat(xs):
   u,s,v = nl.svd(xs,0)
   hatmat = np.dot(u, u.T)
   return hatmat

def jackknife_error(y,xss):
   errs = np.array([0.]*len(xss))
   for ii in range(len(xss)):
       yh,beta,est_w,est_pos = du.ls_est(y,xss[ii],grid)
       active_set = np.nonzero(beta)[0]
       h = get_hatmat(xss[ii][:,active_set])
       hd = np.diag(h)
       errs[ii] = sum((y-np.squeeze(yh))**2/((1-hd)**2))

s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices
kappas = np.arange(0.8,3,.05)

data, bvecs0, bvals = dnd.load_hcp_cso()
bvecs0=bvecs0.T
bval = 2000
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
n= np.shape(bvecs)[0]
xss = du.build_xss(grid,bvecs,kappas)
hatmats = build_hatmats(xss)

# simulation
nits = 100
all_cves0= np.zeros((len(kappas),nits))
all_cves1= np.zeros((len(kappas),nits))
for ii in range(nits):
    true_k = 2
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.ones((true_k,1))/true_k
    true_kappa = 1.6
    true_sigma = 0.1
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    sel_kappa, cves1 = du.cv_sel_params(y1,xss,n,kappas)
    cves0 = jackknife_error(y,xss)
    all_cves0[:,ii] = cves1
    all_cves1[:,ii] = cves0
sum_cves0 = np.sum(all_cves0,axis=1)
sel_kappa0 = kappas[du.rank_simple(sum_cves0)[0]]
sum_cves1 = np.sum(all_cves1,axis=1)
sel_kappa1 = kappas[du.rank_simple(sum_cves1)[0]]



