import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
bvecs = s1.vertices
grid = s2.vertices

true_k = 1
true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
true_w = np.ones((true_k,1))
true_kappa = 1.5
true_sigma = 0.1
y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)

kappas = np.arange(0.5,2,.1)
xss = [0]*len(kappas)
for i in range(len(kappas)):
    kappa = kappas[i]
    xss[i] = du.ste_tan_kappa(np.sqrt(kappa)*grid,bvecs)
cves = [0.]*len(kappas)
for i in range(len(kappas)):
    kappa = kappas[i]
    cves[i] = sum(du.cv_nnls(y1,xss[i],5))
sel_kappa = kappas[du.rank_simple(cves)[0]]
