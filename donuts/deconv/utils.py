import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.optimize as spo
import scipy.spatial.distance as dist
import donuts.emd as emd


def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

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
    y0 : N x 1 numpy array, noiseless signal
    y1 : N x 1 numpy array, noisy signal
    """
    x = ste_tan_kappa(fibers, bvecs)
    n = np.shape(bvecs)[0]
    y0 = np.dot(x,weights)
    raw_err = sigma*np.random.normal(0,1,(n,2))
    raw_err[:,0] = raw_err[:,0] + np.squeeze(y0)
    y1 = norms(raw_err).reshape(-1,1)
    return y0,y1

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

def ls_est(y,xs,grid):
    """ Fits the SFM using NNLS

    Parameters
    ----------
    y : n x 1 numpy array, signal
    xs : n x p numpy array, design matrix
    grid : p x 3 numpy array, candidate directions which were used to generate xs
    
    Outputs
    -------
    yh : n x 1 numpy array, predicted signal
    beta : p x 1 numpy array, the regression coefficients
    est_w : ? x 1 numpy array, the nonnegative entries of beta in descending order
    est_pos : ? x 3 numpy array, the points in grid corresponding to est_w
    """
    beta = spo.nnls(xs,np.squeeze(y))[0]
    est_pos = grid[np.squeeze(np.nonzero(beta)),:]
    est_w = beta[np.nonzero(beta)]
    o = rank_simple(-est_w)
    est_w = est_w[o]
    est_pos = est_pos[o,:]
    yh = np.dot(xs,beta).reshape(-1,1)
    return yh, beta, est_pos, est_w

def sym_emd(true_pos, true_w, est_pos,est_w):
    """ Computes the EMD between two kappa-weighted FODFS

    Parameters
    ----------
    true_pos : K1 x 3 numpy array, ground truth set of directions,
               where each direction is weighted by the square root of its estimated kappa
    true_w   : K1 x 1 numpy array, weights of each fiber.  These will be normalized automatically to sum to 1
    est_pos  : K2 x 3 numpy array, estimated set of directions,
               where each direction is weighted by the square root of its estimated kappa
    est_w    : K1 x 1 numpy array, estimated weights of each fiber.  These will be normalized to sum to 1
    
    Outputs
    -------
    ee       : Earthmover distance, a number from 0 to 4
    """
    true_w = true_w/sum(true_w)
    est_w = est_w/sum(est_w)
    d1 = dist.cdist(true_pos,est_pos)
    d2 = dist.cdist(true_pos,-est_pos)
    dm = np.minimum(d1,d2).ravel()
    ee = emd.emd(list(true_w.ravel()), list(est_w.ravel()), dm)
    return ee

def rand_ortho(k):
    # returns a random orthogonal matrix of size k x k
    a = np.random.normal(0,1,(k,k))
    u, s, v = nla.svd(a)
    return u


def build_xss(grid,bvecs,kappas):
    """ Generates a list of Steksjal-Tanner signal design matrices
        for kappa in kappas
    
    Parameters
    ----------
    grid: M x 3 numpy array of unit vector fiber directions
    bvecs: N x 3 numpy array of unit vectors
      corresponding to DWI measurement directions
    kappas: K-length list of kappas

    Returns
    -------
    xss : K-length list
      xss[i] : N x M numpy array, columns are ST kernel signals
    """
    xss = [0]*len(kappas)
    for i in range(len(kappas)):
        kappa = kappas[i]
        xss[i] = ste_tan_kappa(sqrt(kappa)*grid, bvecs)
    return xss

def cv_sel_params(y,xss,k_folds,params):
    """ Selects model parameters for a single observation using cross-validation

    Parameters
    ----------
    y : n x 1 numpy array, signal
    xss : K-length list of n x p numpy array, design matrix
    k_folds : number of cross-validation folds
    params : K-length list of parameters corresponding to xss
    
    Outputs
    -------
    sel_param: param corresponding to lowest cv error
    cves : K-length list of cv error for each xss
    """
    K = len(kappas)
    cves = [0.]*K
    for i in range(K):
        cves[i] = sum(cv_nnls(y,xss[i],k_folds))
    sel_param = params[rank_simple(cves)[0]]
    return sel_param, cves
