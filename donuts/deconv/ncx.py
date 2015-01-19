import numpy as np
import scipy as sp
import scipy.stats as spst
import scipy.special as sps
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.optimize as spo

# tested in du
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

# tested in du    
def column(x):
    """ Utility function: converts a np array to column vector

    Parameters
    ----------
    x : np array or scalar

    Returns
    -------
    x : np array, column vector

    """
    return np.reshape(x,(-1,1))

# test written
def numderiv(f,x,delta):
    """ Utility function: computes a numerical derivative

    Parameters
    ----------
    f : function with one argument
    x: scalar or np array, points to evaluate f'
    delta: small number 

    Returns
    -------
    ans: np array

    """
    return (f(x+delta)-f(x))/delta

# test written
def numderiv2(f,x,delta):
    """ Utility function: computes a numerical second derivative

    Parameters
    ----------
    f : function with one argument
    x: scalar or np array, points to evaluate f'
    delta: small number 

    Returns
    -------
    ans: np array

    """
    return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)

# test written
def mean_ncx(df,mu0,sigma=1.0):
    """ Approx. mean of a noncentral chi variable
        the norm of N(mu, sigma^2 I_df)
        (the square root of an ncx2 variable)

    Parameters
    ----------
    df: degrees of freedom
    mu: norm of the location parameter of the multivariate normal
    sigma: the marginal standard deviation

    Returns
    -------
    ans: expected norm of the multivariate normal

    """
    nc = (mu0/sigma)**2
    mu = df + nc
    sig2 = 2*df + 4*nc
    the_mean = np.sqrt(mu) - sig2/(8* np.power(mu,1.5))
    return the_mean*sigma

# test written
def ncxloss_gauss(x,df,sigma=1.0): 
    """gaussian approximation to -ncx log likelihood
    
    Parameters:
    -----------
    x: observed value of X in data, a noncentral chi squared variable
    df: known degrees of freedom of X
    sigma: noise level

    Output:
    -------
    ff : function with one argument
         Inputs: mu, square root of noncentrality
         Outputs: val, negative log likelihood of mu
                  der, derivative with respect to mu
    """
    s2 = sigma**2
    s4 = sigma**4
    def ff(mu):
        nc = mu**2
        vv = 2*s4*df+4*s2*nc
        numer = nc+s2*df-x
        val= .5*np.log(2*np.pi*(vv)) + (numer)**2/(2*vv)
        ncx2der= 2*s2/vv + numer/vv - 2*s2*(numer/vv)**2
        der = 2*mu*ncx2der
        ncx2der2 = - 8*(s4 + s2*numer)/(vv**2) + 1/vv + 16*(s2**2)*(numer)**2/(vv**3)
        der2 = 2*ncx2der + (2*mu)**2*ncx2der2
        return val,der,der2
    return ff

# test written
def ncxloss_mean(x,df):
    """gaussian approximation to -ncx log likelihood without variance term
    
    Parameters:
    -----------
    x: observed value of X in data, a noncentral chi squared variable
    df: known degrees of freedom of X

    Output:
    -------
    ff : function with one argument
         Inputs: mu, square root of noncentrality
         Outputs: val, negative log likelihood of mu
                  der, derivative with respect to mu
    """
    def ff(mu):
        nc = mu**2
        val= .5*(nc+df-x)**2
        ncx2der= nc+df-x
        der = 2*mu*ncx2der
        return val,der
    return ff

# test written
def ncxloss_true(x,df): 
    """ true ncx loss calculated using spst
        Warning: fails for large and small values

    Parameters:
    -----------
    x: observed value of X in data, a noncentral chi squared variable
    df: known degrees of freedom of X

    Output:
    -------
    ff : function with one argument
         Inputs: mu, square root of noncentrality
         Outputs: val, negative log likelihood of mu
                  der, derivative with respect to mu
    """
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


# test written
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

