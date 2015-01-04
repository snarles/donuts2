import numpy as np
import scipy as sp
import scipy.stats as spst
import scipy.special as sps
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.optimize as spo

def scalarize(x):
    x = np.atleast_1d(x)
    if len(x)==1:
        return x[0]
    else:
        return x
    
def column(x):
    return np.reshape(x,(-1,1))

def numderiv(f,x,delta):
    return (f(x+delta)-f(x))/delta

def mean_ncx(df,nc0,sigma=1.0): # approximate mean
    nc = nc0/sigma
    mu = nc0+df
    mu = df + nc
    sig2 = 2*df + 4*nc
    the_mean = np.sqrt(mu) - sig2/(8* np.power(mu,1.5))
    return the_mean*sigma

def ncxloss_gauss(x,df): # gaussian approximation to -ncx log likelihood
    def ff(mu):
        nc = mu**2
        val= .5*np.log(2*np.pi*(2*df+4*nc)) + (nc+df-x)**2/(4*df+8*nc)
        ncx2der= 1/(df + 2*nc) + 2*(nc + df-x)/(4*df + 8*nc) - 8*((nc+df-x)/(4*df+8*nc))**2
        der = 2*mu*ncx2der
        return val,der
    return ff

def ncxloss_mean(x,df): # gaussian approximation to -ncx log likelihood without variance term
    def ff(mu):
        nc = mu**2
        val= .5*(nc+df-x)**2
        ncx2der= nc+df-x
        der = 2*mu*ncx2der
        return val,der
    return ff

def ncxloss_true(x,df): # true ncx loss calculated using spst
    val0 = -spst.chi2.logpdf(x,df)
    def ff(mu):
        nc = mu**2
        def calcval(ncc):
            val = np.zeros(len(ncc))
            val[ncc !=0] = -spst.ncx2.logpdf(x,df,ncc[ncc!=0])
            val[ncc ==0] = val0
            return val
        val = calcval(np.atleast_1d(nc))
        dval = calcval(np.atleast_1d(nc + 1e-3))
        ncx2der = 1e3*(dval-val)
        der = 2*mu*ncx2der
        return scalarize(val),scalarize(der)
    return ff

