# carries out CV kappa selection on subset of data

print 'Hello'

import sys
print sys.argv[0]
barg = int(sys.argv[1])
partarg = int(sys.argv[2])


import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import dipy.data as dpd

s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s1.vertices
kappas = np.arange(0.8,3,.05)

data, bvecs0, bvals = dnd.load_hcp_cso()
bvecs0=bvecs0.T
bval = barg
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
xss = du.build_xss(grid,bvecs,kappas)
n= np.shape(bvecs)[0]
    
# synthetic data

nvox = 100
nits = 100
all_cves = np.zeros((len(kappas),nits))
strname = "synth_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
for ii in range(nits):
    print strname+" "+str(ii)
    true_k = 3
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    true_w = np.array([1.,0.,0.])
    true_kappa = 2.0
    true_sigma = 0.1
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    sel_kappa, cves = du.cv_sel_params(y1,xss,n,kappas)
    all_cves[:,ii] = cves    
np.save(strname,all_cves)
