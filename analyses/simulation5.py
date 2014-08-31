# multi-kappa fitting

import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
grid = du.sph_lattice(10,2)


#data, bvecs0, bvals = dnd.load_hcp_cso()
data, bvecs0, bvals = dnd.load_hcp_cc()
bvecs0=bvecs0.T
bval = 2000
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
n= np.shape(bvecs)[0]
xs = du.ste_tan_kappa(grid, bvecs)

# simulation
true_k = 2
true_pos = np.sqrt(true_kappa)*du.normalize_rows(np.random.normal(0,1,(true_k,3)))
true_w = np.ones((true_k,1))/true_k
true_kappa = 1.6
true_sigma = 0.1
y0, y1 = du.simulate_signal_kappa(true_pos,true_w,bvecs,true_sigma)
yh, beta, est_pos, est_w = du.ls_est(y1,xs,grid)
