# Just a script to test things out

import numpy as np
import scipy as sp
import scipy.optimize as spo

def fullfact(levels):
    """ Creates a full factorial design matrix

    Parameters
    ----------
    levels : list of K integers specifying number of levels of each factor

    Returns
    -------
    x : array with K columns, full factorial matrix with levels starting at 0
    """
    x = np.arange(levels[0]).reshape(levels[0],1)
    if len(levels) > 1:
        for i in range(1,len(levels)):
            x2 = np.kron(x, np.ones((levels[i],1)))
            x3 = np.arange(levels[i]).reshape(levels[i],1)
            x4 = np.kron(np.ones((np.shape(x)[0],1)),x3)
            x = np.hstack([x2,x4])
    return x

def norms(x) :
    # computes the norms of the rows of x
    nms = np.sum(np.abs(x)**2,axis=-1)**(1./2)
    return nms

def normalize_rows(x):
    # normalizes the rows of x
    n = np.shape(x)[0]
    p = np.shape(x)[1]
    nms = norms(x).reshape(-1,1)
    x = np.multiply(x, 1./np.tile(nms,(1,p)))
    return x
    

def sph_lattice(resolution, radius):
    """ Creates a ? x 3 array of points *inside* a sphere
    
    Parameters
    ----------
    resolution: determines the minimum distance between points
      as radius/resolution
    radius: radius of the sphere
    
    Returns
    -------
    x : ? x 3 array of points
    """
    k = 2*resolution + 1
    x = fullfact([k,k,k])/resolution - 1
    nms = np.sum(np.abs(x)**2,axis=-1)**(1./2)
    x = x[np.nonzero(nms <= 1)[0],:]
    x = radius * x
    return x 

def ste_tan_kappa(grid, bvecs):
    """ Generates the Steksjal-Tanner signal
        for fibers oriented with directions and kappa determined by
        grid, when measure in directions specified by bvecs.
        Note: kappa will be norm of the vector in grid squared.
    
    Parameters
    ----------
    grid: M x 3 numpy array of fiber directions with length
      equal to square root of kappa
    bvecs: N x 3 numpy array of unit vectors
      corresponding to DWI measurement directions
    
    Returns
    -------
    x : N x M numpy array, columns are ST kernel signals
    """
    x = np.exp(-np.dot(grid, bvecs.T)**2).T
    return x

def simulate_signal_kappa(fibers, weights, bvecs, sigma):
    """ Simulates (Rician) noisy and noiseless signal from a voxel
        with fiber directions specified by fibers,
        fiber kappas specified by root-norm of fibers,
        noise scaling specified by sigma, and 
        measurement directions bvecs
    
    Parameters
    ----------
    fibers: K x 3 numpy array of fiber directions with length
      equal to square root of kappa
    weights: K x 1 numpy array of fiber weights
    bvecs: N x 3 numpy array of unit vectors
      corresponding to DWI measurement directions
    
    Returns
    -------
    signals : 2-length list
    signals[0] : N x 1 numpy array, noiseless signal
    signals[1] : N x 1 numpy array, noisy signal
    """
    x = ste_tan_kappa(fibers, bvecs)
    n = np.shape(bvecs)[0]
    y0 = np.dot(x,weights)
    raw_err = sigma*np.random.normal(0,1,(n,2))
    raw_err[:,0] = raw_err[:,0] + np.squeeze(y0)
    y1 = norms(raw_err).reshape(-1,1)
    return [y0,y1]

def cv_nnls(y,xs,k_folds):
    """ Computes cross-validation error of regression y on xs

    Parameters
    ----------
    y : n x 1 numpy array, signal
    xs : n x p numpy array, design matrix
    k_folds : number of cross-validation folds
    
    Outputs
    -------
    cve : cv error for the k_folds folds
    """
    n = len(y)
    rp = np.random.permutation(n)/float(n)
    cve = np.zeros(k_folds)
    for i in range(k_folds):
        filt_te = np.logical_and(rp >= (float(i)/k_folds), rp < ((float(i)+1)/k_folds))
        y_tr = y[np.nonzero(np.logical_not(filt_te))]
        y_te = y[np.nonzero(filt_te)]
        xs_tr = xs[np.nonzero(np.logical_not(filt_te))]
        xs_te = xs[np.nonzero(filt_te)]
        beta = spo.nnls(xs_tr,np.squeeze(y_tr))[0]
        yh = np.dot(xs_te, beta)
        cve[i] = sum((yh - np.squeeze(y_te))**2)
    return cve

# setup bvecs and grid; using random points for now, but to be replaced by other code
grid = normalize_rows(np.random.normal(0,1,(10000,3)))
bvecs = normalize_rows(np.random.normal(0,1,(n,3)))

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

import dipy.data as dpd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices

