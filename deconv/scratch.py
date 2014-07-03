# Just a script to test things out

import numpy as np
import scipy as sp

def fullfact(levels):
    """ Creates a full factorial design matrix

    Parameters
    ----------
    levels : list of integers specifying number of levels of each factor
    """
    x = np.arange(levels[0]).reshape(levels[0],1)
    if len(levels) > 1:
        for i in range(1,len(levels)):
            x2 = np.kron(x, np.ones((levels[i],1)))
            x3 = np.arange(levels[i]).reshape(levels[i],1)
            x4 = np.kron(np.ones((np.shape(x)[0],1)),x3)
            x = np.hstack([x2,x4])
    return x

def sph_lattice(resolution, radius):
    """ Creates a N x 3 array of points *inside* a sphere
    
    Parameters
    ----------
    resolution: determines the minimum distance between points
      as radius/resolution
    radius: radius of the sphere
    """
