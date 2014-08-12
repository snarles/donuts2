import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices
kappas = np.arange(0.8,3,.05)

#data, bvecs0, bvals = dnd.load_hcp_cso()
data, bvecs0, bvals = dnd.load_hcp_cc()
bvecs0=bvecs0.T
bval = 2000
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
xss = du.build_xss(grid,bvecs,kappas)
n= np.shape(bvecs)[0]

# simulation
nits = 100
all_cves= np.zeros((len(kappas),nits))
for ii in range(nits):
    true_k = 2
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.ones((true_k,1))/true_k
    true_kappa = 1.6
    true_sigma = 0.1
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    sel_kappa, cves = du.cv_sel_params(y1,xss,10,kappas)
    all_cves[:,ii] = cves
sum_cves = np.sum(all_cves,axis=1)
sel_kappa = kappas[du.rank_simple(sum_cves)[0]]
sel_xs = xss[du.rank_simple(sum_cves)[0]]
tsses=[0.]*nits
sses = [0.]*nits
sumbs=[0.]*nits
for ii in range(nits):
    true_k = 3
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.ones((true_k,1))/true_k
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    yh, beta, est_pos, est_w = du.ls_est(y1,sel_xs,grid)
    sumbs[ii]=sum(beta)
    sses[ii] = sum((yh-y1)**2)
    tsses[ii]=sum((y1-y0)**2)
tsigma_h = np.sqrt(np.mean(tsses)/n)
sigma_h = np.sqrt(np.mean(sses)/n)

# real data

sel_inds = [[5,5],[5,15],[5,25],[5,35],[15,5],[15,15],[15,25],[15,35],[25,5],[25,15],[25,25],[25,35],[35,5],[35,15],[35,25],[35,35]]
nits = len(sel_inds)
all_cves = np.zeros((len(kappas),nits))
for ii in range(nits):
    y1 = data[sel_inds[ii][0],sel_inds[ii][1],0,idx]
    sel_kappa, cves = du.cv_sel_params(y1,xss,20,kappas)
    all_cves[:,ii] = cves
sses = [0.]*nits
sumbs=[0.]*nits
sum_cves = np.sum(all_cves,axis=1)
sel_kappa = kappas[du.rank_simple(sum_cves)[0]]
sel_xs = xss[du.rank_simple(sum_cves)[0]]
for ii in range(len(sel_inds)):
    y1 = data[sel_inds[ii][0],sel_inds[ii][1],0,idx]    
    yh, beta, est_pos, est_w = du.ls_est(y1,sel_xs,grid)
    sumbs[ii]=sum(beta)
    sses[ii] = sum((np.squeeze(yh)-y1)**2)
sigma_h = np.sqrt(np.mean(sses)/n)

    
