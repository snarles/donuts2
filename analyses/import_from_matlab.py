# this file was used to create data files in the repo
# do not run the file

import numpy as np
import scipy.io as so
import donuts.deconv.navigator as dcn

datapath = '/biac4/wandell/data/snarles/'
result0 = np.load(datapath+'cc1000.npy')
result = np.load(datapath+'bvecs1000.npy')


result = so.loadmat(datapath+'realdata6_py.mat')
dcn.writeraw1(datapath,'frk1_cso',result['bvals'],result['bvecs_s1'],result['datas_s1'])
dcn.writeraw1(datapath,'frk2_cso',result['bvals'],result['bvecs_s2'],result['datas_s2'])
