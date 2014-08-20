# test isotropic component

import numpy as np
import donuts.deconv.utils as du
import dipy.data as dpd
import donuts.data as dnd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices
kappas = np.arange(0.8,3,.05)

data, bvecs0, bvals = dnd.load_hcp_cso2()
bvecs0=bvecs0.T
bval = barg
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
xss = du.build_xss(grid,bvecs,kappas)
n= np.shape(bvecs)[0]
all_cvess = [0]*16

strheads = ["s3iso_1","s2iso_2","s3iso_3"]
min_mins = [0]*3

for vn in range(1,4):
    for partarg in range(1,17):
        strname = strheads[vn]+"_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
        all_cvess[partarg-1] = np.load(strname)
    all_cves = np.hstack(all_cvess)
    sum_cves = np.sum(all_cves,axis=1)
    sel_kappa = kappas[du.rank_simple(sum_cves)[0]]
    sel_xs = xss[du.rank_simple(sum_cves)[0]]
    min_mins[vn-1] = min(sum_cves)

    nvox = np.shape(data)[0]
    sses = [0.]*nvox
    sumbs=[0.]*nvox
    for ii in range(nvox):
        y1 = data[ii,idx]    
        yh, beta, est_pos, est_w = du.ls_est(y1,sel_xs,grid)
        sumbs[ii]=sum(beta)
        sses[ii] = sum((np.squeeze(yh)-y1)**2)
    sigma_h = np.sqrt(np.mean(sses)/n)

 


