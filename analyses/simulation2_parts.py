# carries out CV kappa selection on subset of data

import sys
barg = int(sys.argv[0])
partarg = int(sys.argv[1])

import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import dipy.data as dpd
strname = "output_b"+str(barg)+"_part_"+str(partarg)+".npy"

s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s2.vertices
kappas = np.arange(0.8,3,.05)

data, bvecs0, bvals = dnd.load_hcp_cso()
bvecs0=bvecs0.T
bval = barg
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
xss = du.build_xss(grid,bvecs,kappas)
n= np.shape(bvecs)[0]

# real data

sel_inds = du.fullfact([40,40])
nvox = len(sel_inds)
nits = 100
inds = range((partarg-1)*nits,min(nvox,partarg*nits))
all_cves = np.zeros((len(kappas),nits))
for ii in range(nits):
    y1 = data[sel_inds[inds[ii]][0],sel_inds[inds[ii]][1],0,idx]
    sel_kappa, cves = du.cv_sel_params(y1,xss,20,kappas)
    all_cves[:,ii] = cves

np.save(strname,all_cves)

    
# synthetic data

sel_inds = du.fullfact([40,40])
nvox = len(sel_inds)
nits = 100
inds = range((partarg-1)*nits,min(nvox,partarg*nits))
all_cves = np.zeros((len(kappas),nits))
partarg = partarg+1
for ii in range(nits):
    true_k = 3
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.array([0.8,0.1,0.1])
    true_kappa = 1.6
    true_sigma = 0.1
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    sel_kappa, cves = du.cv_sel_params(y1,xss,20,kappas)
    all_cves[:,ii] = cves
strname = "synth_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
np.save(strname,all_cves)
