
MLE test for noncentral chi squared


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
    os.chdir("..")
    import donuts.deconv.utils as du
    import donuts.deconv.ncx as ncx


    def numderiv2(f,x,delta):
        return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)
    
    def numderiv(f,x,delta):
        return (f(x+delta)-f(x))/delta
    
    def ncx2loss_gauss(x,df): 
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
        def ff(theta):
            nc = theta[0]
            s2 = theta[1]
            s4 = s2**2
            vv = 2*s4*df+4*s2*nc
            numer = nc+s2*df-x
            val= .5*np.log(2*np.pi*(vv)) + (numer)**2/(2*vv)
            nc_der= 2*s2/vv + numer/vv - 2*s2*(numer/vv)**2
            nc_der2 = - 8*(s4 + s2*numer)/(vv**2) + 1/vv + 16*(s2**2)*(numer)**2/(vv**3)
            return val,der,der2
        return ff


    


    nc = 10000
    df = 10
    n = 100
    x = spst.ncx2.rvs(df,nc,size=n)
    def likelihood(nc):
        return sum(logncx2pdf_x(x,df,nc))


    ncs = np.arange(0.0,nc*2,nc*0.1)
    lks = np.array([likelihood(nch) for nch in ncs])
    plt.scatter(ncs,lks)
    plt.scatter(nc,likelihood(nc),color="green")
    imax = np.where(lks==max(lks))[0][0]
    plt.scatter(ncs[imax],lks[imax],color="red")
    plt.show()


    




    39



Inflection point finding test


    def numderiv2(f,x,delta):
        return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)
    
    df = 10
    x = 1000.0
    
    def ff(mu):
        return -logncx2pdf_nc(x,df,mu**2)
    def f2(mu):
        return numderiv2(ff,mu,1e-3)
    
    mus = np.arange(0.08,10.0,.01)
    y = logncx2pdf_nc(x,df,mus**2)
    d2y = f2(mus)
    
    #plt.scatter(mus,y)
    #plt.scatter(mus,d2y)
    #plt.show()
    
    muinf = spo.newton(f2,1e-2)[0]
    mus = np.arange(0.0,5*muinf,muinf*0.01)
    y = ff(mus)
    plt.scatter(mus,y)
    plt.show()


    df = 10
    x = 1000.0
    def ff(mu):
        return -logncx2pdf_nc(x,df,mu**2)
    cff = convex_nc_loss(x,df)
    mus = np.arange(-20.0,20.0,.1)
    plt.scatter(mus,ff(mus))
    plt.scatter(mus,cff(mus),color="red")
    plt.show()


    def ff(mu):
        return -logncx2pdf_nc(x,df,mu**2)
    def f2(mu):
        return numderiv2(ff,mu,1e-3) - 1e-2
    oguess = df/2
    muinf = spo.newton(f2,oguess)[0]
    while np.absolute(muinf) > df/2:
        oguess = oguess*.9
        muinf = spo.newton(f2,oguess)[0]
    muinf=np.absolute(muinf)
    val = ff(muinf)
    dval = numderiv(ff,muinf,1e-3)
    d2val = 1e-2
    muinf
    mus = np.arange(-100,100,1.0)
    plt.scatter(mus,ff(mus))
    plt.scatter(muinf,ff(muinf),color="red")
    plt.show()
    muinf




    1.802605832650553




    muinf = spo.newton(f2,df/3)[0]
    muinf




    1.8026058332239596



MLE using convex loss


    n = 1000
    df = 10
    mu = 2.0
    x = spst.ncx2.rvs(df,mu**2,size=n)
    ls=[convex_nc_loss(xx,df) for xx in x]
    def likelihood(mu):
        return sum([ll(mu) for ll in ls])


    mus = np.arange(1e-3,mu*2,mu*0.01)
    lks = np.array([likelihood(muh) for muh in mus])
    plt.scatter(mus,lks)
    plt.scatter(mu,likelihood(mu),color="green")
    imax = np.where(lks==min(lks))[0][0]
    plt.scatter(mus[imax],lks[imax],color="red")
    plt.show()


    
