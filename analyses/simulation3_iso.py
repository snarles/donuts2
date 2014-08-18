# tests isotropic vs anisotropic component
# carries out CV kappa selection on subset of data


def cv_sel_params_center(y,xss,k_folds,params):
    K = len(params)
    cves = [0.]*K
    n = len(y)
    rp = np.random.permutation(n)/float(n)
    for j in range(K):
        xs = xss[j]
        cve = np.zeros(k_folds)
        for i in range(k_folds):
            filt_te = np.logical_and(rp >= (float(i)/k_folds), rp < ((float(i)+1)/k_folds))
            y_tr = y[np.nonzero(np.logical_not(filt_te))]
            mu = np.mean(y.tr)
            y_te = y[np.nonzero(filt_te)]
            xs_tr = xs[np.nonzero(np.logical_not(filt_te))]
            xs_te = xs[np.nonzero(filt_te)]
            beta = spo.nnls(xs_tr - mu,np.squeeze(y_tr-mu))[0]
            yh = np.dot(xs_te, beta) + mu
            cve[i] = sum((yh - np.squeeze(y_te))**2)
        cves[j] = sum(cve)
    sel_param = params[rank_simple(cves)[0]]
    return sel_param, cves


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
xss_1 = [np.hstack([np.ones((n,1)),xs]) for xs in xss]
n= np.shape(bvecs)[0]



    
# real data

nvox = np.shape(data)[0]
nits = 10
lrange = (partarg-1)*nits
urange = min(partarg*nits,nvox)
iis = range(lrange,urange)
all_cves = np.zeros((len(kappas),len(iis)))
strname = "s3iso_1_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
for ii in iis:
    if printarg==1:
        print strname+" "+str(ii)
    y1 = data[ii,idx]
    sel_kappa, cves = du.cv_sel_params(y1,xss,n,kappas)
    all_cves[:,ii-lrange] = cves    
np.save(strname,all_cves)

all_cves = np.zeros((len(kappas),len(iis)))
strname = "s2iso_2_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
for ii in iis:
    if printarg==1:
        print strname+" "+str(ii)
    y1 = data[ii,idx]
    sel_kappa, cves = du.cv_sel_params(y1,xss_1,n,kappas)
    all_cves[:,ii-lrange] = cves    
np.save(strname,all_cves)

all_cves = np.zeros((len(kappas),len(iis)))
strname = "s3iso_3_output_b"+str(barg)+"_part_"+str(partarg)+".npy"
for ii in iis:
    if printarg==1:
        print strname+" "+str(ii)
    y1 = data[ii,idx]
    sel_kappa, cves = cv_sel_params_center(y1,xss,n,kappas)
    all_cves[:,ii-lrange] = cves    
np.save(strname,all_cves)
