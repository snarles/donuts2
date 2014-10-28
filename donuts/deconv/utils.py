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
    sigma: noise level
    
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
    cve : sum of squares cv error for the k_folds folds
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
        if n < np.shape(xs)[0]:
            beta = spo.nnls(np.vstack([xs_tr,xs[n+1,:]]),np.squeeze(np.vstack([y_tr,0])))[0]
        else:
            beta = spo.nnls(xs_tr,np.squeeze(y_tr))[0]
        yh = np.dot(xs_te, beta)
        cve[i] = sum((yh - np.squeeze(y_te))**2)
    return cve

def ls_est(y,xs,grid):
    """ Fits the SFM using NNLS

    Parameters
    ----------
    y : n x 1 numpy array, signal
    xs : n (or n+1) x p numpy array, design matrix
    grid : p x 3 numpy array, candidate directions which were used to generate xs
    
    Outputs
    -------
    yh : n x 1 numpy array, predicted signal
    beta : p x 1 numpy array, the regression coefficients
    est_w : ? x 1 numpy array, the nonnegative entries of beta in descending order
    est_pos : ? x 3 numpy array, the points in grid corresponding to est_w
    """
    n = np.shape(y)[0]
    if n < np.shape(xs)[0]:
        beta = spo.nnls(xs,np.squeeze(np.vstack([y,0])))[0]
        yh = np.dot(xs[0:n,:],beta).reshape(-1,1)
    else:
        beta = spo.nnls(xs,np.squeeze(y))[0]
        yh = np.dot(xs,beta).reshape(-1,1)
    est_pos = grid[np.squeeze(np.nonzero(beta)),:]
    est_w = beta[np.nonzero(beta)]
    o = rank_simple(-est_w)
    est_w = est_w[o]
    est_pos = est_pos[o,:]
    return yh, beta, est_pos, est_w

def lasso_est(y,xs,grid,l1p):
    yh, beta, est_pos, est_w = ls_est(np.vstack([y,0]),np.vstack([xs,l1p*np.ones((1,np.shape(xs)[1]))]),grid)
    yh = yh[:-1,]
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

def arc_emd(true_pos,true_w,est_pos,est_w):
    """ Computes the EMD between two unit length fODFS using arc-length distance

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
    ee       : Earthmover distance, a number from 0 to pi/2
    """
    true_pos = normalize_rows(true_pos)
    est_pos = normalize_rows(est_pos)
    true_w = true_w/sum(true_w)
    est_w = est_w/sum(est_w)
    dm = arcdist(true_pos,est_pos).ravel()
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
        xss[i] = ste_tan_kappa(np.sqrt(kappa)*grid, bvecs)
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
    K = len(params)
    cves = [0.]*K
    n = len(y)
    rp = np.random.permutation(n)/float(n)
    for j in range(K):
        xs = xss[j]
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
        cves[j] = sum(cve)
    sel_param = params[rank_simple(cves)[0]]
    return sel_param, cves


def cv_sel_params_center(y,xss,k_folds,params):
    """ Selects model parameters for a single observation using cross-validation using mean-centering

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
    K = len(params)
    cves = [0.]*K
    n = len(y)
    rp = np.random.permutation(n)/float(n)
    for j in range(K):
        xs = xss[j]
        xs = np.array([xx - np.mean(xx) for xx in xs.T]).T
        cve = np.zeros(k_folds)
        for i in range(k_folds):
            filt_te = np.logical_and(rp >= (float(i)/k_folds), rp < ((float(i)+1)/k_folds))
            y_tr = y[np.nonzero(np.logical_not(filt_te))]
            mu = np.mean(y_tr)
            y_te = y[np.nonzero(filt_te)]
            xs_tr = xs[np.nonzero(np.logical_not(filt_te))]
            xs_te = xs[np.nonzero(filt_te)]
            beta = spo.nnls(xs_tr,np.squeeze(y_tr-mu))[0]
            yh = np.dot(xs_te, beta) + mu
            cve[i] = sum((yh - np.squeeze(y_te))**2)
        cves[j] = sum(cve)
    sel_param = params[rank_simple(cves)[0]]
    return sel_param, cves

def arcdist(xx,yy):
    """ Computes pairwise arc-distance matrix

    Parameters
    ----------
    xx : a x 3 numpy array
    yy : b x 3 numpy array

    Outputs
    -------
    dd: a x b numpy array
    """
    dd = np.arcsin(np.absolute(np.dot(xx,yy.T)))
    dd = np.nan_to_num(dd)
    return dd

# generates k random unit vectors with minimum arc length gap
def randvecsgap(k,gap):
    if gap==0:
        ans = np.random.randn(k,3)
        ans = normalize_rows(ans)
        return ans
    ans = np.zeros((k,3))
    flag = True
    idxs = np.zeros(k,dtype=bool)
    while sum(idxs) < k:
        idx = np.random.randint(k)
        idxs[idx] = False
        v = np.random.normal(0,1,(1,3))
        v = v/nla.norm(v)
        if sum(idxs) > 0:
            dd = arcdist(v, ans[idxs,:])
            if np.amin(np.squeeze(dd)) > gap:
                ans[idx,:] =  v
                idxs[idx] = True
        else:
            ans[idx,:] =  v
            idxs[idx] = True
    return ans

def bsel_test(y, xs, grid):
    """ Fits the SFM using NNLS then uses backwards pruning

    Parameters
    ----------
    y : n x 1 numpy array, signal
    xs : n (or n+1) x p numpy array, design matrix
    grid : p x 3 numpy array, candidate directions which were used to generate xs
    
    Outputs
    -------
    yhs : n x K numpy array, predicted signal for K = 1...
    betas : p x K numpy array, the regression coefficients for K=...
    est_s : 1xK list, contains (est_pos,est_w) pairs
    sses : 1xK, errors for backwards selection
    """
    n = np.shape(y)[0]
    yh, beta, est_pos, est_w = du.ls_est(y,xs,grid)
    sparsity = len(np.nonzero(beta)[0])
    est_s = [0]*sparsity
    est_s[0] = [[est_w,est_pos]]
    yhs = np.zeros((n,sparsity))
    yhs[:,0] = yh
    betas = np.zeros((n,sparsity))
    betas[:,0] = beta
    sparss = sparsity-np.array(range(sparsity))
    sses = [0.]*sparsity
    sses[0] = sum((y-yh)**2)
    min_sses = np.array([100.]*sparsity)
    min_sses[0] = sum((y1-yh)**2)
    active_sets = [0.]*sparsity
    active_set = np.nonzero(beta)[0]
    for i in range(sparsity):        
        active_sets[i] = active_set
        if i < sparsity-1:
            sse = np.array([0.]*len(active_set))
            for j in range(len(active_set)):
                a_new = np.delete(active_set,j,0)
                yh_t = du.ls_est(y,xs[:,a_new],grid[a_new,:])[0]
                sse[j] = sum((y-yh_t)**2)
            sses[i]=sse
            min_sses[i+1] = min(sse)
            j_sel = du.rank_simple(sse)[0]
            active_set = np.delete(active_set, j_sel,0)
            yh, beta, est_pos, est_w = du.ls_est(y,sel_xs[:,active_set],grid[active_set,:])
            yhs[:,i] = yh
            est_w = beta
            est_pos=grid[active_set,:]
            est_s[i] = [[est_w,est_pos]]
    spars = [len(v) for v in active_sets]
    return errs, min_sses, spars,sumbs

