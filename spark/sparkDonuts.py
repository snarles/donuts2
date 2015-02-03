
# coding: utf-8

# In[3]:

import os
import numpy as np
import numpy.random as npr
import numpy.linalg as nla
import scipy as sp
import scipy.stats as spst
import scipy.special as sps
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.optimize as spo
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pyemd import emd


# In[4]:

def norms(x) :
    """ computes the norms of the rows of x """
    nms = np.sum(np.abs(x)**2,axis=-1)**(1./2)
    return nms

def normalize_rows(x):
    """ normalizes the rows of x """
    n = np.shape(x)[0]
    p = np.shape(x)[1]
    nms = norms(x).reshape(-1,1)
    x = np.multiply(x, 1./np.tile(nms,(1,p)))
    return x

def arcdist(xx, yy):
    """ Computes pairwise arc-distance matrix"""
    dm = np.absolute(np.dot(xx,yy.T))
    dm[dm > .99999999999999999] = .99999999999999999
    dd = np.arccos(dm)
    return dd

def divsum(x):
    return x/sum(x)

def arc_emd(x1, w1, x2, w2):
    x1 = x1[w1 > 0, :]
    w1 = w1[w1 > 0]
    x2 = x2[w2 > 0, :]
    w2 = w2[w2 > 0]
    w1 = divsum(w1)
    w2 = divsum(w2)
    arg1 = np.hstack([w1, 0*w2])
    arg2 = np.hstack([0*w1, w2])
    arg3 = arcdist(np.vstack([x1, x2]), np.vstack([x1, x2]))
    return emd(arg1, arg2, arg3)


# In[8]:

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

def rand_ortho(k):
    """ returns a random orthogonal matrix of size k x k """
    a = np.random.normal(0,1,(k,k))
    u, s, v = nla.svd(a)
    return u

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

def georandsphere(n,k):
    temp = [0]*k
    grid0 = geosphere(n)
    for ii in range(k):
        temp[ii] = np.dot(grid0,rand_ortho(3))
    ans = np.vstack(temp)
    return ans


# In[24]:

def rvs_ncx2(df,mu,sz=1,sigma = 1.0):
    """ Generate noncentral chi-squared random variates with mu, sigma parameterization
    i.e. the squared norm of a multivariate normal

    Parameters
    ----------
    df : degrees of freedom
    mu: the norm of the mean of the multivariate normal
    sz: the number of variates to generate
    sigma: the marginal standard deviation of the multivariate normal

    Returns
    -------
    ans: np array
    """
    mu = np.atleast_1d(mu)
    if len(mu) ==1:
        ans = (mu*np.ones(sz) + sigma*npr.normal(0,1,sz))**2 + (sigma**2)*spst.chi2.rvs(df-1,size=sz)
        return ans
    else:
        ans = (np.squeeze(mu) + sigma*npr.normal(0,1,len(mu)))**2 + (sigma**2)*spst.chi2.rvs(df-1,size=len(mu))
        return ans


# In[14]:

ncases = 10
k = 3
emd_test_cases = [(ii, {'x1': normalize_rows(npr.normal(0, 1, (k, 3))), 'w1': divsum(npr.exponential(1, k)),                'x2': normalize_rows(npr.normal(0, 1, (k, 3))), 'w2': divsum(npr.exponential(1, k))}) for ii in range(ncases)]


# In[15]:

[arc_emd(**case[1]) for case in emd_test_cases]


# In[11]:

sgrid = geosphere(5)
sgrid = sgrid[sgrid[:,2] >0, :]
np.shape(sgrid)


# In[18]:

sgrid0 = np.dot(sgrid, rand_ortho(3))
bvecs = geosphere(4)


# In[67]:

kappa0 = 2.0
kappa = kappa0


# In[68]:

amat0 = ste_tan_kappa(np.sqrt(kappa)*sgrid0, bvecs)
pp0 = np.shape(amat0)[1]
amat = ste_tan_kappa(np.sqrt(kappa)*sgrid, bvecs)
pp = np.shape(amat)[1]


# In[77]:

k0 = 3
df = 64
sigma0 = 0.2/np.sqrt(df)
w0 = np.zeros(pp0)
w0[npr.randint(0, pp0, k0)] = 1.0/k0
mu = np.squeeze(np.dot(amat0, w0))
y = rvs_ncx2(df, mu, sigma = sigma0)


# In[78]:

bt_nnls = spo.nnls(amat, np.sqrt(y))[0]
arc_emd(sgrid0, w0, sgrid, bt_nnls)


# In[79]:

bt_nnls2 = spo.nnls(amat, np.sqrt(y- df*(sigma0**2)))[0]
arc_emd(sgrid0, w0, sgrid, bt_nnls2)


# In[82]:

#plt.scatter(mu, np.sqrt(y))


# In[83]:

#plt.scatter(mu, np.sqrt(y - df*(sigma0**2)))


# In[ ]:



