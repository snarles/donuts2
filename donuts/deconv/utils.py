import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.optimize as spo
import scipy.spatial.distance as dist
import scipy.stats as spst
import donuts.emd as emd

# test written
def scalarize(x):
    """ Utility function: converts a np array or scalar to scalar if len==1

    Parameters
    ----------
    x : np array or scalar

    Returns
    -------
    x : np array or scalar

    """
    x = np.atleast_1d(x)
    if len(x)==1:
        return x[0]
    else:
        return x

# test written    
def column(x):
    """ Utility function: converts a np array to column vector

    Parameters
    ----------
    x : np array or scalar

    Returns
    -------
    x : np array, column vector

    """
    return np.reshape(np.array(x),(-1,1))

# test written
def rank_simple(vector):
    """ Utility function: returns ranks of a vector

    Parameters
    ----------
    vector: np array

    Returns
    -------
    vector: np array

    """
    return sorted(range(len(vector)), key=vector.__getitem__)

# test written
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

# test written
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

# test written
def ordered_partitions(n,k):
    """ Forms all k-length nonnegative integer partitions of n (where ordering matters)

    Parameters
    ----------
    n: integer > 0, total number of elements
    k: integer > 0, max number of sets in partition

    Returns
    -------
    ans: np array, ?? x k, each row is a partition

    """
    if k==1:
        return n*np.ones((1,1))
    subparts = [0]*(n+1)
    for ii in range(n+1):
        temp = ordered_partitions(n-ii,k-1)
        temp_p = np.shape(temp)[0]
        subparts[ii] = np.hstack([ii*np.ones((temp_p,1)),temp])
    return np.vstack(subparts)

# skipped test
def norms(x) :
    """ computes the norms of the rows of x """
    nms = np.sum(np.abs(x)**2,axis=-1)**(1./2)
    return nms

# test written
def normalize_rows(x):
    """ normalizes the rows of x """
    n = np.shape(x)[0]
    p = np.shape(x)[1]
    nms = norms(x).reshape(-1,1)
    x = np.multiply(x, 1./np.tile(nms,(1,p)))
    return x

# test written    
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

# test written
def simulate_signal_kappa(fibers, weights, bvecs, sigma, df=2):
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
    df: degrees of freedom
    
    Returns
    -------
    y0 : N x 1 numpy array, noiseless signal
    y1 : N x 1 numpy array, noisy signal
    """
    x = ste_tan_kappa(fibers, bvecs)
    n = np.shape(bvecs)[0]
    y0 = np.dot(x,weights)
    y0sq = (y0/sigma)**2
    y1sq = spst.ncx2.rvs(df,y0sq)
    y1 = np.sqrt(y1sq)*sigma
    return column(y0),column(y1)

# test written
def rand_ortho(k):
    """ returns a random orthogonal matrix of size k x k """
    a = np.random.normal(0,1,(k,k))
    u, s, v = nla.svd(a)
    return u

# test written
def randvecsgap(k,gap):
    """ generates k random unit vectors with minimum arc length gap

    Parameters
    ----------
    k: number of vectors to be generate
    gap: float > 0, minimum arc distance gap

    Outputs
    -------
    ans: k x 3 numpy array consisting of unit vectors

    """
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

# test written
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
    dm = np.absolute(np.dot(xx,yy.T))
    dm[dm > .99999999999999999] = .99999999999999999
    dd = np.arccos(dm)
    return dd

# test written
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
    true_pos = true_pos[np.squeeze(true_w) > 0,:]
    true_w = true_w[np.squeeze(true_w) > 0]
    est_pos = est_pos[np.squeeze(est_w) > 0,:]
    est_w = est_w[np.squeeze(est_w) > 0]
    dm = arcdist(true_pos,est_pos).ravel()
    ee = emd.emd(list(true_w.ravel()), list(est_w.ravel()), dm)
    return ee

# test written
def geosphere(n):
    """ returns a ??x3 spherical design

    Parameters
    ----------
    n: number of subdivisions

    Outputs
    -------
    ans: ?? x 3 numpy array consisting of unit vectors
         symmetric about the z-axis

    """
    # set up icosahedron
    v = np.zeros((3,12))
    v[:,0] = [0,0,1]
    v[:,11] = [0,0,-1]
    seq1 = 2.0*np.arange(1,6,1.0)*np.pi/5
    seq2 = seq1 + np.pi/5
    v[:,1:6] = 2.0/np.sqrt(5) * np.vstack([np.cos(seq1),np.sin(seq1),0.5*np.ones(5)])
    v[:,6:11] = 2.0/np.sqrt(5) * np.vstack([np.cos(seq2),np.sin(seq2),-.5*np.ones(5)])
    edges = [0]*30
    for ii in range(5):
        edges[ii] = (v[:,0],v[:,1+ii])
        edges[2*(ii+1)+8] = (v[:,1+ii],v[:,6+ii])
        edges[25+ii] = (v[:,11],v[:,6+ii])
    for ii in range(4):
        edges[ii+5] = (v[:,1+ii],v[:,2+ii])
        edges[ii+20] = (v[:,6+ii],v[:,7+ii])
        edges[2*(ii+1)+9] = (v[:,2+ii],v[:,6+ii])
    edges[9] = (v[:,5],v[:,1])
    edges[19] = (v[:,1],v[:,10])
    edges[24] = (v[:,10],v[:,6])

    faces = [0]*20
    for ii in range(4):
        faces[ii] = (v[:,0],v[:,1+ii],v[:,2+ii])
        faces[15+ii] = (v[:,11],v[:,6+ii],v[:,7+ii])
        faces[2*(ii+1)+3] = (v[:,1+ii],v[:,6+ii],v[:,2+ii])
        faces[2*(ii+1)+4] = (v[:,6+ii],v[:,2+ii],v[:,7+ii])
    faces[4] = (v[:,0],v[:,5],v[:,1])
    faces[19] = (v[:,11],v[:,10],v[:,6])
    faces[13] = (v[:,5],v[:,10],v[:,1])
    faces[14] = (v[:,10],v[:,1],v[:,6])
    # interpolate
    v_final = [v]
    pp = 12+30*(n-1)+10*(n-1)*(n-2)
    if n > 1:
        seq = np.arange(1,n,1.0)
        mat = np.vstack([seq/n,1-(seq/n)])
        v_edges = np.hstack([np.dot(np.vstack([x[0],x[1]]).T,mat) for x in edges])
        v_final = v_final+[v_edges]
    if n > 2:
        mat2 = (1.0/n * (ordered_partitions(n-3,3)+1)).T
        v_faces = np.hstack([np.dot(np.vstack([x[0],x[1],x[2]]).T,mat2) for x in faces])
        v_final = v_final+[v_faces]
    v_f = np.hstack(v_final)
    v_norm = np.vstack([x/nla.norm(x) for x in v_f.T]).T
    return v_norm.T
