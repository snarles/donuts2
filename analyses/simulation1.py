import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices
kappas = np.arange(1.5,4,.1)

data, bvecs0, bvals = dnd.load_hcp_cso()
bvecs0=bvecs0.T
bval = 3000
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
xss = du.build_xss(grid,bvecs,kappas)
n= np.shape(bvecs)[0]

# simulation
true_k = 3
true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
true_w = np.ones((true_k,1))
true_kappa = 2
true_sigma = 0.2
y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
sel_kappa, cves = du.cv_sel_params(y1,xss,20,kappas)

# real data
kappas = np.arange(1.5,4,.1)
sel_kappa, cves = du.cv_sel_params(data[5,3,0,idx],xss,20,kappas)
sel_kappa

