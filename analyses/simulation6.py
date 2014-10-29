# peak finding test

import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices
grid = grid[grid[:,1] >= 0,:]
grid = du.normalize_rows(grid)
dm = du.arcdist(grid,grid)
kappas = np.arange(1.5,4,.1)
bvecs = s1.vertices
n= np.shape(bvecs)[0]
true_kappa = 2
xs = du.ste_tan_kappa(np.sqrt(true_kappa)*grid, bvecs)

# simulation
true_k = 3
true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
true_w = np.ones((true_k,1))
true_sigma = 0.1
y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
yh, beta, est_pos, est_w = du.ls_est(y1,xs,grid)
err = du.arc_emd(true_pos,true_w,est_pos,est_w)
yhs, betas, est_s, sses= du.bsel_nnls(y1, xs, grid)
errs = [du.arc_emd(true_pos,true_w,v[1],v[0]) for v in est_s]
est_pos2,est_w2 = du.peak_1(beta,grid,dm,0.1,0.05)
err1 = du.arc_emd(true_pos,true_w,est_pos2,est_w2)
