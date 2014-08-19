# backwards selection method for sparse beta

import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
import matplotlib.pyplot as plt

s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices

data, bvecs0, bvals = dnd.load_hcp_cso()
bvecs0=bvecs0.T
bval = 2000
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
n= np.shape(bvecs)[0]


def bsel_test(true_kappa, true_pos, true_w, bvecs, true_sigma, sel_xs, grid):
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    yh, beta, est_pos, est_w = du.ls_est(y1,sel_xs,grid)
    err = du.sym_emd(true_pos,true_w,est_pos,est_w)
    sparsity = len(np.nonzero(beta)[0])
    sparss = sparsity-np.array(range(sparsity))
    sses = [0.]*sparsity
    min_sses = np.array([100.]*sparsity)
    min_sses[0] = sum((y1-yh)**2)
    sumbs = min_sses
    sumbs[0] = sum(est_w)
    active_sets = [0.]*sparsity
    errs = np.array([100.]*sparsity)
    active_set = np.nonzero(beta)[0]
    for i in range(sparsity):
        errs[i] = du.sym_emd(true_pos,true_w,est_pos,est_w)
        active_sets[i] = active_set
        if i < sparsity-1:
            sse = np.array([0.]*len(active_set))
            for j in range(len(active_set)):
                a_new = np.delete(active_set,j,0)
                yh_t = du.ls_est(y1,sel_xs[:,a_new],grid[a_new,:])[0]
                sse[j] = sum((y1-yh_t)**2)
            sses[i]=sse
            min_sses[i+1] = min(sse)
            j_sel = du.rank_simple(sse)[0]
            active_set = np.delete(active_set, j_sel,0)
            yh, beta, est_pos, est_w = du.ls_est(y1,sel_xs[:,active_set],grid[active_set,:])
            est_w = beta
            sumbs[i+1] = sum(est_w)
            est_pos=grid[active_set,:]
    spars = [len(v) for v in active_sets]
    return errs, min_sses, spars,sumbs

def bsel_gen_test(true_k, true_kappa, true_sigma):
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.ones((true_k,1))/true_k
    sel_xs = du.ste_tan_kappa(np.sqrt(true_kappa)*grid, bvecs)
    errs,min_sses,spars,sumbs = bsel_test(true_kappa,true_pos,true_w,bvecs,true_sigma,sel_xs,grid)

errs,min_sses,spars,sumbs = bsel_gen_test(true_k,true_kappa,true_sigma)
