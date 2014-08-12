# carries out CV kappa selection on subset of data

import sys
print sys.argv[0]
barg = int(sys.argv[1])
partarg = int(sys.argv[2])
printarg = int(sys.argv[3])

import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import dipy.data as dpd

s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
grid = s1.vertices
kappas = np.arange(0.8,3,.05)

data, bvecs0, bvals = dnd.load_hcp_cc()
bvecs0=bvecs0.T
bval = barg
idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
bvecs = bvecs0[idx,:]
xss = du.build_xss(grid,bvecs,kappas)
n= np.shape(bvecs)[0]
    
# real data

nvox = np.shape(data)[0]
nits = 100
lrange = (partarg-1)*nits
urange = min(partarg*nits,nvox)
iis = range(lrange,urange)
all_cves = np.zeros((len(kappas),len(iis)))
strname = "s2cc_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
for ii in iis:
    if printarg==1:
        print strname+" "+str(ii)
    y1 = data[ii,idx]
    sel_kappa, cves = du.cv_sel_params(y1,xss,n,kappas)
    all_cves[:,ii-lrange] = cves    
np.save(strname,all_cves)
