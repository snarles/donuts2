import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.optimize as spo
import scipy.spatial.distance as dist
import donuts.emd as emd


from numpy.linalg import lapack_lite
lapack_routine = lapack_lite.dgesv
def faster_inverse(A):
    """ inverts the matrices a = A[i,:,:]
    the following code was written by Carl F 
    see http://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
    """
    b = np.identity(A.shape[2], dtype=A.dtype)

    n_eq = A.shape[1]
    n_rhs = A.shape[2]
    pivots = np.zeros(n_eq, np.intc)
    identity  = np.eye(n_eq)
    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = np.zeros(n_eq, np.intc)
        results = lapack_lite.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        if results['info'] > 0:
            raise LinAlgError('Singular matrix')
        return b

    return np.array([lapack_inverse(a) for a in A])

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

def inds_fullfact(levels, col_inds, values):
    """ Returns a list of indices for which fullfact(levels)[inds, col_inds[i]] = values[i]

    Parameters
    ----------
    levels : list of K integers specifying number of levels of each factor
    col_inds: indices of columns 0...(K-1)
    values: values, value[i] between 0...levels[col_inds[i]]

    Returns
    -------
    inds : list of ? integers, indices where selected values are taken
    """
    nfacts = len(levels)
    vals = np.array([-1]*nfacts,dtype=int)
    vals[col_inds] = values
    tracker = nfacts-1
    cur_arr = np.array([0],dtype=int)
    flag = True
    temp = [1] + levels[::-1]
    temp = temp[0:-1]
    pd = np.cumprod(temp,dtype=int)[::-1]
    while tracker > -1:
        if vals[tracker]==-1:
            cur_arr = np.hstack([cur_arr + pd[tracker]*ii for ii in range(levels[tracker])])
        if vals[tracker] > 0:
            cur_arr = cur_arr + pd[tracker]*vals[tracker]
        tracker = tracker-1
    return cur_arr

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
    dd = np.arccos(np.absolute(np.dot(xx,yy.T)))
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

def bsel_nnls(y, xs, grid):
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
    min_sses : 1xK, errors for backwards selection
    """
    n = np.shape(y)[0]
    pp = np.shape(xs)[1]
    yh, beta, est_pos, est_w = ls_est(y,xs,grid)
    sparsity = len(np.nonzero(beta)[0])
    est_s = [0]*sparsity
    est_s[0] = [est_w,est_pos]
    yhs = np.zeros((n,sparsity))
    yhs[:,0:1] = yh
    betas = np.zeros((pp,sparsity))
    betas[:,0] = beta
    sparss = sparsity-np.array(range(sparsity))
    min_sses = np.array([100.]*sparsity)
    min_sses[0] = sum((y-yh)**2)
    active_sets = [0.]*sparsity
    active_set = np.nonzero(beta)[0]
    for i in range(1,sparsity):        
        active_sets[i] = active_set
        sse = np.array([0.]*len(active_set))
        for j in range(len(active_set)):
            a_new = np.delete(active_set,j,0)
            yh_t = ls_est(y,xs[:,a_new],grid[a_new,:])[0]
            sse[j] = sum((y-yh_t)**2)
        min_sses[i] = min(sse)
        j_sel = rank_simple(sse)[0]
        active_set = np.delete(active_set, j_sel,0)
        yh, beta, est_pos, est_w = ls_est(y,xs[:,active_set],grid[active_set,:])
        yhs[:,i] = yh[:,0]
        est_w = beta
        est_pos=grid[active_set,:]
        est_s[i] = [est_w,est_pos]
    return yhs, betas, est_s, min_sses

def peak_1(beta,grid,dm,gap,thres):
    """ Peak-finding algorithm applied to fODF with distance matrix dm

    Parameters
    ----------
    beta : pp x 1 numpy array, coefficients of fODF
    grid: pp x 3 grid points, directions of fODF
    dm: pp x pp distance matrix
    gap: scalar minimal gap distance parameter
    thres: scalar minimal peak threshold parameter
    
    Outputs
    -------
    est_pos: Kx3 estimated position
    est_w: Kx1 weights
    """
    beta2 = beta/sum(beta)
    pp = np.shape(grid)[0]
    sparsity = len(np.nonzero(beta)[0])
    est_pos = np.zeros((sparsity,3))
    est_w = np.zeros((sparsity,1))
    flag = True
    count = 0
    while flag:
        j_sel = rank_simple(beta2)[-1]
        newmask = (dm[j_sel,:] < gap)
        coef = sum(beta2[newmask])
        if coef > thres:
            est_pos[count,:] = grid[j_sel,:]
            est_w[count] = coef
            beta2[newmask] = 0
            count = count+1
        else:
            flag = False
        if max(beta2)==0:
            flag = False
    est_pos = est_pos[range(count),:]
    est_w = est_w[range(count),:]
    est_w = est_w/sum(est_w)
    return est_pos, est_w

def cap_sample(v,rad,res):
    """ Supplies unit vectors which form a grid on a spherical cap

    Parameters
    ----------
    v : (3,) vector
    rad: radius of cap, from 0 to 1 
    res: resolution of cap, the number of points will be approx. pi * res^2

    Outputs
    -------
    vs: ?x3 unit vectors forming a cap around v
    """
    a = np.identity(3)
    a[:,0] = v
    q,r = np.linalg.qr(a)
    q[:,0] = v
    kk = 2*res + 1
    x = fullfact([kk,kk])/res - 1
    nms = norms(x)
    x = x[nms < 1,:]
    x = x*rad
    x = np.hstack([1+0*x[:,0:1],x])
    x = normalize_rows(x)
    vs = np.dot(x,q.T)
    return vs

def build_bcn_xss(est_pos,bvecs,kappa,rad,res):
    """ Build the xss list for best_combo_nnls using peaks found by peakfinding algorithm

    Parameters
    ----------
    est_pos : (K,3) peaks
    bvecs: (n,3) measurement vectors
    kappa: parameter for Ste-Tan eq
    rad: radius of cones around peaks
    res: resolution of cap, the number of points will be approx. pi * res^2

    Outputs
    -------
    xss : K-length list, xss[i] is (n,?) design matrix
    grids: K-length list, corresponding positions
    """
    K = np.shape(est_pos)[0]
    xss = [0]*K
    grids = [0]*K
    for ii in range(K):
        grids[ii] = cap_sample(est_pos[ii,:],rad,res)
        xss[ii] = ste_tan_kappa(np.sqrt(kappa) * grids[ii],bvecs)
    return grids,xss

def best_combo_nnls(y,xss,grids):
    """ Best K-subset nnls taking 1 (or 0) directions from each set of directions xss[0],..,xss[K]

    Parameters
    ----------
    y : (n,1) data
    xss : K-length list, xss[i] is (n,pps[i]) design matrix
    grids: K-length list, corresponding positions

    Outputs
    -------
    sses: prod(pp[i]+1)-list, sum of square errors for combinations
    choices: assignments of directions
    est_pos: direction of best subset
    est_w: weights of best subset
    """
    pps = np.array([np.shape(xs)[2] for xs in xss])
    choices = fullfact(pps)
    nchoice = np.shape(choices)[0]
    K = length(xss)
    grams = np.zeros((nchoice,K**2))
    # fill out the gram matrix
    for ii in range(0,K):
        for jj in range(ii,K):
            gram0 = np.dot(xss[ii].T,xss[jj])
            
