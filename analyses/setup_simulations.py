# Run this file before running any simulations
# Usage: python setup_simulations.py [data_path]
# all simulation data will be saved in data path

import sys
datapath = sys.argv[1]
f = open('datapath.txt','w')
f.write(datapath)
f.close()

import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import dipy.data as dpd
import time

today = time.strftime("%d-%m-%Y")

f = open(datapath + 'index.txt','w')
f.write('INDEX\n')
f.close()

f = open(datapath + 'index.txt','a')

for bval in [1000,2000,3000]:
    data, bvecs0, bvals = dnd.load_hcp_cso2()
    bvecs0=bvecs0.T
    idx = np.squeeze(np.nonzero(np.logical_and(bvals > bval-20, bvals < bval+20)))
    bvecs = bvecs0[idx,:]
    np.save(datapath + 'bvecs'+str(bval),bvecs)
    np.save(datapath + 'cso'+str(bval),data[:,idx])
    f.write('name:cso'+str(bval)+' bvecs:bvecs'+str(bval)+' type:raw date:'+today+'\n')
    data, bvecs0, bvals = dnd.load_hcp_cc()
    np.save(datapath + 'cc'+str(bval),data[:,idx])
    f.write('name:cc'+str(bval)+' bvecs:bvecs'+str(bval)+' type:raw date:'+today+'\n')

f.close()


