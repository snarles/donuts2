import numpy as np
import numpy.testing as npt
import scipy.spatial.distance as dist

import donuts.emd as emd


def test_emd():
    xx = np.array([[1,0,0], [0,1,0], [0,0,1]])
    cost = []

    for this_x in xx:
        for this_y in xx:
            cost.append(np.sqrt(sum((this_x - this_y)**2)))

    ee = emd.emd([0, 1, 0], [1, 0, 0], cost)
    npt.assert_almost_equal(ee, np.sqrt(2), decimal=5)

    # What about different locations? 
    xx =  np.array([[1,0,0], [0,1,0], [0,0,1]])
    yy = np.array([[1,0,0], [0,1,0], [0,0,1], [np.sqrt(2), np.sqrt(2), 0]])
    cost = dist.cdist(xx, yy).ravel()
    ee = emd.emd([0, 1, 0], [1, 0, 0, 0], cost)
    npt.assert_almost_equal(ee, np.sqrt(2), decimal=5)
