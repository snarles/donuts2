# this file was used to create data files in the repo
# do not run the file

import numpy as np
import scipy.io as so
import 

datapath = '/biac4/wandell/data/snarles/'
result0 = np.load(datapath+'cc1000.npy')
result = np.load(datapath+'bvecs1000.npy')


result = so.loadmat(datapath+'realdata7.mat')
bv11 = result['bvecss'][1,1].T
cc_coords = result['coords'][:,0:506].T
cc_datas = np.hstack(result['datas'][1,0:506,1]).T
