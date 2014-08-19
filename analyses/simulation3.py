# backwards selection method for sparse beta

import sys
bval = int(sys.argv[1])
true_k = int(sys.argv[2])
scale_p = int(sys.argv[3])
true_kappa = float(sys.argv[4])
true_sigma = float(sys.argv[5])
partarg = int(sys.argv[6])


strname = "sim3_b"+sys.argv[1]+"_k"+sys.argv[2]+"_s"+sys.argv[3]+"_kp"+sys.argv[4]+"_sg"+sys.argv[5]+"_"+sys.argv[6]
print strname


import numpy as np
import numpy.random as nr
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
import matplotlib.pyplot as plt

s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices

data, bvecs0, bvals = dnd.load_hcp_cso()
bvecs0=bvecs0.T
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
    sumbs = [0.]*sparsity
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

def bsel_gen_test(true_k, scale_p, true_kappa, true_sigma, grid, bvecs, sel_xs):
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.random.dirichlet([scale_p]*true_k,1).T
    errs,min_sses,spars,sumbs = bsel_test(true_kappa,true_pos,true_w,bvecs,true_sigma,sel_xs,grid)
    return errs, min_sses,spars,sumbs,true_pos,true_w

def no_bksel(otpts):
    errs = [otpt[0,0] for otpt in otpts]
    spars=[otpt[0,2] for otpt in otpts]
    return np.mean(errs),np.mean(spars)

def thres_bksel(otpts,thres):
    errs = [0.]*len(otpts)
    spars = [0.]*len(otpts)
    for ii in range(len(otpts)):
        otpt = otpts[ii]
        indd=0
        errs[ii] = otpt[0,0]
        sses = otpt[:,1]
        sse_diff = sses[1:]-sses[:-1]
        ll=np.nonzero(sse_diff > thres)[0]
        if len(ll) > 0:
           indd = ll[0]
        errs[ii] = otpt[indd,0]
        spars[ii] = otpt[indd,2]
    return np.mean(errs),np.mean(spars)

def thres_bksel_m(otpts, thress):
    for thres in thress:
        me,ms = thres_bksel(otpts,thres)
        print (thres, me,ms)

def gen_otpts(true_k,scale_p,true_kappa,true_sigma,nits):
    otpts = [0]*nits
    true_poss = [0]*nits
    true_ws = [0]*nits
    sel_xs = du.ste_tan_kappa(np.sqrt(true_kappa)*grid, bvecs)
    for ii in range(nits):
        errs,min_sses,spars,sumbs,true_pos,true_w = bsel_gen_test(true_k,scale_p,true_kappa,true_sigma,grid,bvecs,sel_xs)
        true_poss[ii] = true_pos
        true_ws[ii] = true_w
        otpts[ii] = np.array(zip(errs,min_sses,spars,sumbs))
    return otpts

