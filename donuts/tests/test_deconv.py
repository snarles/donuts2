import numpy as np
import scipy as sp
import scipy.optimize as spo
import donuts.deconv.utils as du

import dipy.data as dpd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices


def test_fullfact():
    a = du.fullfact([3,3,3])


def blah():
    # setup bvecs and grid; using random points for now, but to be replaced by other code
    grid = du.normalize_rows(np.random.normal(0,1,(10000,3)))
    bvecs = du.normalize_rows(np.random.normal(0,1,(n,3)))
    true_kappa = 1.5
    true_pos = normalize_rows(np.random.normal(0,1,(3,3)))
    true_w = np.array([1,1,1]).reshape((-1,1))
    res = simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,0.1)
    y0=res[0]
    y1=res[1]

    # test if NNLS recovers the correct positions for noiseless data
    kappa=1.5
    xs = ste_tan_kappa(np.sqrt(kappa)*grid,bvecs)
    beta = spo.nnls(xs,np.squeeze(y0))[0]
    est_pos = grid[np.nonzero(beta),:]
    est_w = beta[np.nonzero(beta)]

