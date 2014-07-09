import numpy as np
import scipy as sp
import scipy.optimize as spo
import donuts.deconv.utils as du
import numpy.testing as npt
import scipy.spatial.distance as spd

import dipy.data as dpd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
bvecs = s1.vertices
grid = s2.vertices

def test_fullfact():
    a = du.fullfact([3,3,3])
    npt.assert_almost_equal(np.size(a),27)

def test_normalize_rows():
    a = du.normalize_rows(np.random.normal(0,1,(10000,3)))
    npt.assert_almost_equal(sum(a[1,]**2),1)

def test_simulate_signal_kappa():
    true_kappa = 1.5
    true_pos = bvecs[[1,10,11],]
    true_w = np.array([1,1,1]).reshape((-1,1))
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,0.1)
    # test if NNLS recovers the correct positions for noiseless data
    kappa=1.5
    xs = du.ste_tan_kappa(np.sqrt(kappa)*grid,bvecs)
    beta = spo.nnls(xs,np.squeeze(y0))[0]
    est_pos = grid[np.squeeze(np.nonzero(beta)),:]
    est_w = beta[np.nonzero(beta)]
    d1 = spd.cdist(true_pos,est_pos) 
    d2 = spd.cdist(true_pos,-est_pos) 
    



