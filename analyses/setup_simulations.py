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

data, bvecs0, bvals = dnd.load_hcp_cso2()
bvecs0=bvecs0.T
np.save(datapath + 'bvecs',bvecs0)
np.save(datapath + 'bvals',bvals)
np.save(datapath + 'cso',data)
data, bvecs0, bvals = dnd.load_hcp_cc()
np.save(datapath + 'cc',data)

f = open(datapath + 'index.txt','w')
f.write('[INDEX][DATA:cso:raw][DATA:cc:raw]')
f.close()



