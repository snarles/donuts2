import os
import numpy as np
import scipy as sp
import scipy.stats as spst
import scipy.special as sps
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.optimize as spo
import donuts.deconv.splines as spl

def numderiv2(f,x,delta):
    return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)

def numderiv(f,x,delta):
    return (f(x+delta)-f(x))/delta

def logivy(v,y):
    y = np.atleast_1d(y)
    ans = np.array(y)
    ans[y < 500] = np.log(sps.iv(v,y[y < 500]))
    ans[y >= 500] = y[y >= 500] - np.log(2*np.pi*y[y >= 500])
    return ans

def logncx2pdf_x(x,df,nc): #only x varies
    if nc==0:
        return spst.chi2.logpdf(x,df)
    else:
        return -np.log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc) + logivy((df/2-1),np.sqrt(nc*x))

def logncx2pdf_nc(x,df,nc0): #only nc varies
    nc0 =np.atleast_1d(nc0) 
    nc = np.array(nc0)
    nc[nc0 < 1e-5] = 1e-5
    ans= -np.log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc) + logivy((df/2-1),np.sqrt(nc*x))
    return ans

def convex_nc_loss(x,df):
    def ff(mu):
        return -logncx2pdf_nc(x,df,mu**2)
    def f2(mu):
        return numderiv2(ff,mu,1e-3) - 1e-2
    mugrid = np.arange(0.0,2*df,df*0.01)
    res = np.where(f2(mugrid) < 1e-2)[0]
    if len(res) > 0:
        imin = np.where(f2(mugrid) < 1e-2)[0][-1]
        muinf = mugrid[imin]
    else:
        muinf = 0.0
    val = ff(muinf)
    dval = numderiv(ff,muinf,1e-3)
    d2val = 1e-2
    #print(muinf)
    def cff(mu):
        mu = np.atleast_1d(mu)
        ans = np.array(mu)
        ans[mu > muinf] = -logncx2pdf_nc(x,df,mu[mu > muinf]**2)
        ans[mu <= muinf] = val + (mu[mu <= muinf]-muinf)*dval + .5*d2val*(mu[mu <= muinf]-muinf)**2
        return ans
    return cff

def pruneroutine(v1,v2):
    # finds the minimum convex combination of v1 and v2 so that exactly one element is nonpositive (zero)
    # v1 is nonnegative
    v1 = np.atleast_1d(v1)
    v2 = np.atleast_1d(v2)
    assert min(v1)>=0
    if min(v2) >=0:
        return v2,-1
    else:
        mina = np.array(v2)*0
        mina[v1 != v2]= -v1[v1 != v2]/(v2[v1 != v2]-v1[v1 != v2])
        ans = (1-mina)*v1 + mina*v2
        assert min(ans) >= -1e-15
        mina[v2 >= 0] = 1e99
        mina[v2==v1] = 1e99
        a = min(mina)
        assert a <= 1
        assert a >= 0
        o = np.where(mina == a)[0][0]
        ans = (1-a)*v1 + a*v2
        assert min(ans) >= -1e-15
        ans[ans <0] =0
        return ans,o

def subrefitting(amat,ls,x0,newind): # refit x0 so that grad(x0)=0 where x0 positive
    oldx0 = np.array(x0)
    s = np.zeros(p,dtype=bool)
    s[np.squeeze(x0) > 1e-20] = True
    s[newind] = True
    amat2 = amat[:,s]
    x02 = np.array(x0[s])
    x02 = bfgssolve(amat2,ls,np.array(x02),-1.0)[0]
    oldx02 = np.array(x02)
    x0[~s] = 0.0
    x0[s]=x02
    flag = min(x0) < 0
    x0 = pruneroutine(oldx0,np.array(x0))[0]
    while flag:
        oldx0 = np.array(x0)
        s = np.zeros(p,dtype=bool)
        s[np.squeeze(x0) > 1e-20] = True
        amat2 = amat[:,s]
        x02 = np.array(x0[s])
        x02 = bfgssolve(amat2,ls,np.array(x02),-1.0)[0]
        x0[~s] = 0.0
        x0[s]=x02
        flag = min(x0) < 0
        x0new = np.array(x0)
        #print(min(x0))
        x0 = pruneroutine(oldx0,np.array(x0))[0]
    return x0

def ebp(amat,ls,x0): # ls is a list of loss functions, x0 is initial guess
    x0seq = [np.array(x0)]
    newind = np.where(x0==max(x0))[0]
    p = np.shape(amat)[1]
    flag = True
    count = 0
    while flag:
        count = count + 1
        # **** refitting step ****
        x0 = subrefitting(amat,ls,np.array(x0),newind)
        # next candidate step
        yh = np.dot(amat,x0)
        rawg = np.array([ls[i](yh[i])[1] for i in range(n)])
        g = np.dot(rawg.T,amat)
        if min(g) > -1e-5:
            flag=False
        else:
            newind = np.where(g==min(g))[0][0]
        if count > 1000:
            flag = False
        x0seq = x0seq + [np.array(x0)]
    return x0,x0seq

def bfgssolve(amat,ls,x0,lb=0.0): # use LBFS-G to solve
    def f(x0):
        yh = np.dot(amat,x0)
        return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
    def fprime(x0):
        yh = np.dot(amat,x0)
        rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
        return np.dot(rawg.T,amat)
    bounds = [(lb,100.0)] * len(x0)
    res = spo.fmin_l_bfgs_b(f,np.squeeze(x0),fprime=fprime,bounds=bounds)
    return res
    

def ncxlosses(df,y):
    n = len(y)
    ans = [0.] * n
    for ii in range(n):
        x = y[ii]
        mmax = np.sqrt(x)*3
        mugrid = np.arange(0,mmax,mmax/100)
        clos = convex_nc_loss(n,x)
        pts = clos(mugrid)
        f = spl.convspline(mugrid,pts)
        ans[ii]=f
    return ans
