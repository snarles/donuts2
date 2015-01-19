import os
import numpy as np
import numpy.random as npr
import numpy.linalg as nla
import scipy.stats as spst
import scipy as sp
import scipy.optimize as spo
import scipy.special as sps
import numpy.testing as npt
import scipy.spatial.distance as spd
import donuts.deconv.utils as du
import donuts.deconv.ncx as ncx
import donuts.deconv.splines as spl

#def test_fail():
#    assert 1==2

# -------------------
# donuts.deconv.utils
# -------------------
# has separate test file

# ---------------------
# donuts.deconv.splines
# ---------------------

def test_b4sd():
    x = np.arange(-2,6,0.01)
    y = spl.bs4d(x)
    def f(x):
        return spl.bs4d(x)[0]
    npt.assert_almost_equal(y[1],ncx.numderiv(f,x,1e-8),decimal = 5)
    npt.assert_almost_equal(y[2],ncx.numderiv2(f,x,1e-4),decimal = 2)
    return

def test_splinemat():
    a = spl.splinemat(npr.randint(5,10))
    npt.assert_almost_equal(a[0,0:3],np.array([1,4,1]))
    npt.assert_almost_equal(a[-1,-3:],np.array([1,4,1]))
    return

def test_splinemat2():
    a = spl.splinemat2(npr.randint(5,10))
    npt.assert_almost_equal(a[0,0:3],np.array([6,-12,6]))
    npt.assert_almost_equal(a[-1,-3:],np.array([6,-12,6]))
    return

def test_splinecoef():
    k = npr.randint(3,10)
    v = npr.normal(0,1,k)
    a = spl.splinemat(k)
    b = spl.splinecoef(k)
    bt = np.dot(b,v)
    npt.assert_almost_equal(np.dot(a,bt),v)
    return

def test_genspline():
    k = npr.randint(10,20)
    v = npr.normal(0,1,k)
    a = spl.splinemat(k)
    b = spl.splinecoef(k)
    bt = np.dot(b,v)
    f = spl.genspline(bt,1,0)
    npt.assert_almost_equal([f(x)[0][0] for x in np.arange(1,k,1)],v[1:])
    return

def test_autospline():
    # random cosine function
    kc = 10
    coefs = npr.normal(0,1,kc)
    def f(x):
        exmat = np.dot(du.column(x),du.column(range(kc)).T)
        return np.dot(np.cos(exmat),coefs)
    lb = npr.normal(0,1)
    ub = lb + npr.exponential(2)
    x = np.arange(lb,ub,(ub-lb)/npr.randint(30,70))
    y = f(x)
    f2 = spl.autospline(x,y)
    xgrid = np.arange(x[5],x[-5],(ub-lb)/1000)
    ygrid = f(xgrid)
    yspl = np.array([f2(xx)[0] for xx in xgrid])
    temp = np.cov(ygrid.T,yspl.T)
    npt.assert_almost_equal(temp[1,0]/temp[0,0],1.0,decimal=2)
    #plt.scatter(xgrid,ygrid)
    #plt.scatter(xgrid,yspl,color="red")
    #plt.show()
    return

def test_convspline():
    kc = 10
    cfsa = .1*npr.uniform(-3,3,kc)
    cfsb = npr.uniform(-1,1,kc)
    def lsef(x): # scalar argument
        return np.log(sum(np.exp(cfsa * x + cfsb)))
    lb = npr.uniform(-10,-5)
    ub = npr.uniform(5,10)
    x = np.arange(lb,ub,(ub-lb)/npr.randint(30,70))
    y = np.array([lsef(xx) for xx in x])
    f = spl.convspline(x,y)
    y2 = np.array([f(xx)[0] for xx in x])
    temp = np.cov(y[1:-2].T,y2[1:-2].T)
    npt.assert_almost_equal(temp[1,0]/temp[0,0],1.0,decimal=2)
    #plt.scatter(x,y)
    #plt.scatter(x,y2,color="red")
    #plt.show()
    for ii in range(100):
        p1 = npr.normal(0,1000)
        p2 = npr.normal(0,1000)
        assert f((p1+p2)/2.0)[0] < (f(p1)[0]+f(p2)[0])/2.0
    return

# -----------------
# donuts.deconv.ncx
# -----------------
# has separate test file
