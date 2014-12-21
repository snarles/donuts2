import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.stats as spst
import scipy.special as sps
import numpy.random as npr
import matplotlib.pyplot as plt
import cvxopt as cvx

def bs4d(z): # standard basis function with derivatives
    i1 = np.array((z >= 0) & (z < 1),dtype = float)
    i2 = np.array((z >= 1) & (z < 2),dtype = float)
    i3 = np.array((z >= 2) & (z < 3),dtype = float)
    i4 = np.array((z >= 3) & (z < 4),dtype = float)
    c1 = z**3
    d1 = 3*z**2
    e1 = 6*z
    c2 = -3*z**3 + 12*z**2 - 12*z + 4
    d2 = -9*z**2 + 24*z - 12
    e2 = -18*z + 24
    c3 = 3*(z**3) - 24*(z**2) + 60*z - 44
    d3 = 9*(z**2) - 48*z + 60
    e3 = 18*z - 48
    c4 = (4-z)**3
    d4 = -3*(z-4)**2
    e4 = -6*(z-4)
    val0= i1*c1+i2*c2+i3*c3+i4*c4
    val1= i1*d1+i2*d2+i3*d3+i4*d4
    val2= i1*e1+i2*e2+i3*e3+i4*e4
    return val0,val1,val2

def genspline(bt,scale,shift): # generates a spline from coefficients
    def f(x): # only evaluates at a single point
        z = scale*(x + shift) + 3
        h = np.floor(z)
        val0 = 0.0
        val1 = 0.0
        val2 = 0.0
        zr = z-h
        for j in range(-4,1):
            if (h+j) >=0 and (h+j) < len(bt):
                cf = bt[h+j]
                e0,e1,e2 = bs4d(zr-j)
                val0 += cf*e0
                val1 += cf*e1
                val2 += cf*e2
        return val0,val1*scale,val2*(scale**2)
    return f

def autospline(x,y): # generates a spline from uniform data
    mmax = max(x)
    mmin = min(x)
    nn = len(x)
    scale = (nn-1)/(mmax-mmin)
    shift = -mmin
    bt = np.dot(splinecoef(nn),y)
    return genspline(bt,scale,shift)

def convspline(x,y): # generates a convex spline
    m=len(x)
    mmax = max(x)
    mmin = min(x)
    scale = (m-1)/(mmax-mmin)
    shift = -mmin
    atamat = cvx.matrix(np.dot(splinemat(m).T,splinemat(m))+ 1e-3 * np.diag(np.ones(m+2)))
    ytamat = cvx.matrix(-np.dot(y.T,splinemat(m)))
    hmat = cvx.matrix(-splinemat2(m))
    zerovec = cvx.matrix(np.zeros((m,1)))
    sol = cvx.solvers.qp(atamat,ytamat,hmat,zerovec)
    bt = np.squeeze(sol['x'])
    return genspline(bt,scale,shift)

def splinemat(m):
    n = m+1
    mat= np.diag(1.* np.ones(n+2)) + np.diag(4.*np.ones(n+1),1) + np.diag(1.* np.ones(n),2)
    return mat[:(n-1),:(n+1)]

def splinemat2(m):
    n = m+1
    mat= np.diag(6.* np.ones(n+2)) + np.diag(-12.*np.ones(n+1),1) + np.diag(6.* np.ones(n),2)
    return mat[:(n-1),:(n+1)]

def splinecoef(m):
    return npl.pinv(splinemat(m))
