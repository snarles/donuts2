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

def test_numderiv():
    def f(x):
        return x**2
    x = np.arange(-5,5,0.1)
    npt.assert_almost_equal(2*x,ncx.numderiv(f,x,1e-3),decimal=3)
    return
def test_numderiv2():
    def f(x):
        return x**2
    x = np.arange(-5,5,0.1)
    npt.assert_almost_equal(2+0*x,ncx.numderiv2(f,x,1e-3),decimal=3)
    return

def test_mean_ncx():
    for ii in range(5):
        df = npr.randint(2,20)
        mu = npr.uniform(0,10)
        x = spst.ncx2.rvs(df,mu**2,size=1e6)
        npt.assert_almost_equal(mu*np.mean(np.sqrt(x)),mu*ncx.mean_ncx(df,mu),decimal=1)
    return

def test_ncxloss_gauss():
    # demonstrate approximation to true loss function
    def subroutine(df,mu0):
        x = spst.ncx2.rvs(df,mu0**2)
        mus = np.arange(mu0*0.1,2*mu0,mu0*0.1)
        f0 = ncx.ncxloss_true(x,df)
        f = ncx.ncxloss_gauss(x,df)
        y0= f0(mus)[0]
        y = f(mus)[0]
        return spst.pearsonr(y,y0)[0]
    # moderate accuracy for small arguments
    reps = 50
    pars = zip(npr.randint(2,20,size=reps),npr.uniform(1,10,size=reps))
    rs= np.array([subroutine(df,mu0) for (df,mu0) in pars])
    assert sum(rs < .5) < 10
    # high accuracy for large arguments
    df = npr.randint(5,10)
    x = npr.uniform(70,100)
    mus = np.arange(30,70,1.0)
    f0 = ncx.ncxloss_true(x,df)
    f = ncx.ncxloss_gauss(x,df)
    y0= f0(mus)[0]
    y = f(mus)[0]
    npt.assert_almost_equal(spst.pearsonr(y,y0)[0],1,decimal=3)
    # test derivatives
    df = npr.randint(5,10)
    x = npr.uniform(70,100)
    mus = np.arange(30,70,1.0)
    sigma = npr.uniform(.5,2.0)
    f = ncx.ncxloss_gauss(x,df,sigma)
    def fval(x):
        return f(x)[0]
    npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=3)
    npt.assert_almost_equal(f(mus)[2],ncx.numderiv2(fval,mus,1e-3),decimal=3)
    # demonstrate asymptotic consistency
    df = npr.randint(5,20)
    mu0 = npr.uniform(20,50)
    sigma = npr.uniform(.5,2.0)
    x = ncx.rvs_ncx2(df,mu0,1e4,sigma)
    ls = [ncx.ncxloss_gauss(xx,df,sigma) for xx in x]
    def likelihood(mu):
        return sum(np.array([ll(mu)[0] for ll in ls]))
    lk0 = likelihood(mu0)
    assert lk0 < likelihood(mu0 * .5)
    assert lk0 < likelihood(mu0 * 1.01)
    assert lk0 < likelihood(mu0 * 0.99)
    assert lk0 < likelihood(mu0 * 2)
    assert lk0 < likelihood(mu0 * npr.uniform(1e-2,100))
    return

def test_ncxloss_mean():
    # demonstrate asymptotic consistency
    df = npr.randint(5,20)
    mu0 = npr.uniform(20,50)
    x = spst.ncx2.rvs(df,mu0**2,size=1e5)
    ls = [ncx.ncxloss_mean(xx,df) for xx in x]
    def likelihood(mu):
        return sum(np.array([ll(mu)[0] for ll in ls]))
    lk0 = likelihood(mu0)
    assert lk0 < likelihood(mu0 * 1.1)
    assert lk0 < likelihood(mu0 * 0.9)
    # test derivatives
    f = ls[0]
    mus = np.arange(0.1,2.0,0.1)*mu0
    def fval(x):
        return f(x)[0]
    npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=-3)
    return

def test_ncxloss_true():
    # demonstrate asymptotic consistency
    df = npr.randint(2,4)
    mu0 = npr.uniform(0,2)
    x = spst.ncx2.rvs(df,mu0**2,size=1e4)
    ls = [ncx.ncxloss_true(xx,df) for xx in x]
    def likelihood(mu):
        return sum(np.array([ll(mu)[0] for ll in ls]))
    lk0 = likelihood(mu0)
    assert lk0 < likelihood(mu0 * .5)
    assert lk0 < likelihood(mu0 * 2)
    return

def test_rvs_ncx2():
    for ii in range(10):
        df = npr.randint(1,100)
        sigma = npr.uniform(.1,3)
        mu = npr.uniform(0,10)
        x = ncx.rvs_ncx2(df,mu,1000000,sigma)
        #x = sigma*spst.ncx2.rvs(df,(mu/sigma)**2,size = 100000)
        npt.assert_almost_equal(np.mean(x)/(mu**2 + (sigma**2)*df),1,decimal=2)
        npt.assert_almost_equal(np.var(x)/(4*(sigma**2)*mu**2 + 2*(sigma**4)*df),1,decimal=2)
    return

def test_sph2cart():
    xyz0 = du.geosphere(4)
    rtp = du.cart2sph(xyz0)
    xyz = du.sph2cart(rtp)
    npt.assert_almost_equal(xyz,xyz0,decimal=10)
    return

def test_georandsphere():
    xyz = du.georandsphere(4,5)
    return

def test_rsh_basis():
    sgrid = du.georandsphere(5,8)
    xs = du.rsh_basis(sgrid,4)
    fudge = .5
    npt.assert_almost_equal(fudge*np.dot(xs.T,xs),fudge*np.eye(np.shape(xs)[1]),decimal=2)
    return

def test_randfunc():
    f = du.randfunc(20.0,.5)
    return
