import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
bvecs = s1.vertices
grid = s2.vertices
n= np.shape(bvecs)[0]

true_k = 3
true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
true_w = np.ones((true_k,1))
true_kappa = 2
true_sigma = 0.2
y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)

kappas = np.arange(1.5,4,.1)
xss = [0]*len(kappas)
for i in range(len(kappas)):
    kappa = kappas[i]
    xss[i] = du.ste_tan_kappa(np.sqrt(kappa)*grid,bvecs)
cves = [0.]*len(kappas)
for i in range(len(kappas)):
    kappa = kappas[i]
    cves[i] = sum(du.cv_nnls(y1,xss[i],20))
sel_kappa = kappas[du.rank_simple(cves)[0]]


import donuts.data as dnd
data, bvecs, bvals = dnd.load_hcp_cso()
bvecs=bvecs.T
y = data[2,3,0,:]
kappas = np.arange(.5,4,.1)
xss = [0]*len(kappas)
for i in range(len(kappas)):
    kappa = kappas[i]
    xss[i] = du.ste_tan_kappa(np.sqrt(kappa)*grid,bvecs)
cves = [0.]*len(kappas)
for i in range(len(kappas)):
    kappa = kappas[i]
    cves[i] = sum(du.cv_nnls(y,xss[i],20))
sel_kappa = kappas[du.rank_simple(cves)[0]]
