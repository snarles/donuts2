import numpy as np
import scipy as sp
import scipy.stats as spst
import scipy.special as sps
import numpy.random as npr

def numderiv(f,x,delta):
    return (f(x+delta)-f(x))/delta

def numderiv2(f,x,delta):
    return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)

def maxseq(f): # finds the max of a unimodal function
    # find an upper bound
    diff = 1
    ub=1
    eps=0.1
    while diff > 0:
        ub=2*ub
        diff = f(ub*(1+eps))-f(ub)
    # bisection search
    lb=0
    x = (ub+lb)/2.0
    while (ub-lb)/ub > eps:
        x = (ub+lb)/2.0
        diff = f(x*(1+eps))-f(x)
        if diff < 0:
            ub = x
        if diff >= 0:
            lb = x
    return x

def lbseq(f,x0,eps): # finds lower bound subject to precision
    diff = x0/2
    x = x0-diff
    while abs(f(x0) - f(x)) > eps:
        diff = diff/2
        x = x0-diff
    diff = diff*2
    x = x0-diff
    x = max([0,x])
    x = np.floor(x)
    return x

def ubseq(f,x0,eps): # finds upper bound subject to precision
    diff = (x0 + 100.0)**2
    x=x0 + diff
    while abs(f(x0) - f(x)) > eps:
        diff = diff/2
        x = x0+diff
    diff = diff*2
    x = x0+diff
    x = np.floor(x)
    x = max([x,np.floor(x0)+10.0])
    return x

def boundseq(f,eps): # finds lower and upper bounds
    x0=maxseq(f)
    lb = lbseq(f,x0,eps)
    ub = ubseq(f,x0,eps)
    return lb,ub

def losssq(mu,x):
    return [(mu-x)**2,2*(mu-x),2]

def logsumexp(xs):
    xs = np.array(xs)
    return max(xs) + np.log(sum(np.exp(xs-max(xs))))

def logsumdiff(x,y): # computes log(e^x - e^y) with sign
    sgn = np.sign(x-y)
    m = max([x,y])
    ans = m + np.log(abs(np.exp(x-m)-np.exp(y-m)))
    return sgn,ans

def logncx2(n,nc,x):
    ws = 200.0
    eps = 30.0
    if nc < 1e-20:
        nc = 1e-20
    a = n/2.0
    temp0 = -(nc+x)/2.0 - a*np.log(2.0) + (a-1)*np.log(x)
    
    # adaptive choose sum indices
    def f0(z):
        return -sps.gammaln(a + z) + z*np.log(nc*x/4) - sps.gammaln(z+1)
    lb,ub = boundseq(f0,eps)
    iz0 = np.arange(lb,ub,1.0)
    tempA = -sps.gammaln(a + iz0) + iz0*np.log(nc*x/4)
    s0 = logsumexp(temp0+tempA - sps.gammaln(iz0+1))
    
    def f1(z):
        return -sps.gammaln(a + z) + z*np.log(nc*x/4) - sps.gammaln(z+1) + np.log(z)
    lb,ub = boundseq(f1,eps)
    iz1 = np.arange(lb,ub,1.0)
    tempB = -sps.gammaln(a + iz1) + iz1*np.log(nc*x/4)
    s1 = logsumexp(temp0+tempB - sps.gammaln(iz1))
    
    def f2(z):
        return -sps.gammaln(a + z) + z*np.log(nc*x/4) - sps.gammaln(z+1) + 2.0*np.log(z)
    lb,ub = boundseq(f2,eps)
    iz2 = np.arange(lb,ub,1.0)
    tempC = -sps.gammaln(a + iz2) + iz2*np.log(nc*x/4)
    s2 = logsumexp(temp0+tempC - sps.gammaln(iz2+1)+2*np.log(iz2))
    
    templ = np.log((nc**(-2.0) + nc**(-1.0)))
    l0= s0 # log likelihood
    sgnraw1,lraw1= logsumdiff(s1 - np.log(nc),s0-np.log(2.0)) # derivative of likelihood
    l1= sgnraw1*np.exp(lraw1-l0) # first derivative of log likelihood
    sub1 = logsumexp([-2.0*np.log(nc)+s2,np.log(.25)+s0])
    sgnraw2,lraw2 = logsumdiff(sub1,templ+s1) # second derivative of likelihood
    if sgnraw2 > 0:
        sgn2,ll2 = logsumdiff(lraw2-l0, 2.0*(lraw1-l0))
        l2 = sgn2*np.exp(ll2)
    if sgnraw2 <= 0:
        l2 = -1.0*np.exp(logsumexp([lraw2-l0, 2.0*(lraw1-l0)]))
    return l0,l1,l2

def logncx2prox(n,nc,x): # normal approximation
    mm = n+nc
    vv = 2*n + 4*nc
    l0 = -.5*np.log(2*np.pi*vv) - .5*(x - mm)**2/vv
    return l0

def losschi2(n,mu,x):
    nc = mu**2.0
    l0,l1,l2 = logncx2(n,nc,x)
    return -1.0*np.array([l0,2*mu*l1,2*l1 + ((2*mu)**2)*l2])

def minchi2(n,mu0,x): # finds the min of chi2
#    return 0
#mu0 = ex
#if True:
    mu = mu0
    for i in range(40):
        res = losschi2(n,mu,x)
        #print [res[0],mu]
        mu = mu-res[1]/res[2]
    return mu

def inflchi2(n,mu0,x): # finds the inflection point of chi2
    lb = 0
    ub = mu0
    for i in range(40):
        mu = (lb+ub)/2
        res = losschi2(n,mu,x)[2]
        if res < 0:
            lb = mu
        if res >= 0:
            ub = mu
    return mu

def genconvexchi2(n,x): # returns a function which is the convex relaxation of losschi2
    mustar = minchi2(n,n/2.0,x)
    muinf = inflchi2(n,mustar,x) + 0.01
    res = losschi2(n,muinf,x)
    def convloss(mu):
        if mu > muinf:
            return losschi2(n,mu,x)
        else:
            return [res[0] + res[1]*(mu-muinf)+res[2]*(mu-muinf)**2,res[1] + res[2]*(mu-muinf),res[2]]
    return convloss
