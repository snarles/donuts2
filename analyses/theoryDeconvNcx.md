
Does the correct loss function improve estimation of directions or kappa
parameter?


    import os
    import numpy as np
    import scipy.optimize as spo
    import scipy.stats as spst
    import numpy.random as npr
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    
    os.chdir("..")
    import donuts.deconv.utils as du
    import donuts.deconv.ncx as nc
    import donuts.deconv.splines as spl

    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)



    reload(spl)
    reload(nc)
    reload(du)




    <module 'donuts.deconv.utils' from 'donuts/deconv/utils.pyc'>



Generate the signal


    # setting up measurement vectors and fitting vectors
    kappa = 1.5
    bvecs = np.sqrt(kappa) * du.geosphere(5)
    n = np.shape(bvecs)[0]
    print("Number of measurement directions is "+str(n))
    sgrid = du.geosphere(8)
    sgrid = sgrid[sgrid[:,2] >= 0,:]
    pp = np.shape(sgrid)[0]
    print("Number of candidate directions is "+str(pp))
    # do you want plots?
    plotsignal = True

    Number of measurement directions is 252
    Number of candidate directions is 341



    # randomly generate parameters
    true_k = 2
    true_vs = du.normalize_rows(npr.normal(0,1,(true_k,3)))
    true_vs[:,2] = np.absolute(true_vs[:,2])
    true_ws = 0.5*np.ones((true_k,1))/true_k
    true_sigma = 1.0
    df = 10
    y0,y1 = du.simulate_signal_kappa(true_vs,true_ws,bvecs,true_sigma,df)
    # plot the noiseless signal
    if plotsignal:
        zz = np.squeeze(y1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(zz*bvecs[:,0],zz*bvecs[:,1],zz*bvecs[:,2])
        plt.show()

Fit the signal, knwoing the true kappa and sigma


    xs = du.ste_tan_kappa(sgrid,bvecs)
    assert np.shape(xs)==(n,pp)
    bt_nnls = spo.nnls(xs,np.squeeze(y1))[0]
    y1sq = (y1/true_sigma)**2
    ls = nc.ncxlosses(df,y1sq)
    bt_rician = nc.bfgssolve(xs,ls,bt_nnls/true_sigma)[0]


    yh[ii][0]




    0.031514958900447455




    ii=1
    ls[ii](yh[ii])




    (array([ 2.97697276]), array([ 0.01652466]), 0.04939097894971281)




    ls[ii](0.14720411)




    (3.8231838894472125, -0.68037099160249359, 0.27739427727240562)




    def likelihood(yh):
        return sum([ls[ii](yh[ii])[0] for ii in range(len(ls))])
    likelihood(10*np.squeeze(y0/true_sigma)**2)




    735.26349554358615




    




    1.6184449800648406




    n = 20
    p = 500
    amat = np.absolute(npr.normal(0,1,(n,p)))
    bt0 = np.zeros((p,1))
    bt0[:2] = 1
    df = 10
    mu = np.dot(amat,bt0)
    ysq = spst.ncx2.rvs(df,mu**2)
    ls = nc.ncxlosses(df,ysq)
    
    def f(x0):
        yh = np.dot(amat,x0)
        return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
    def fprime(x0):
        yh = np.dot(amat,x0)
        rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
        return np.dot(rawg.T,amat)


    bt = spo.nnls(amat,np.squeeze(np.sqrt(ysq)))[0]
    #print(f(bt))
    res = nc.bfgssolve(amat,ls,np.array(bt),0.0)
    x0 = res[0]
    (f(x0),sum(x0 > 0))




    (64.041474329709757, 16)



## Development of NCX loss


    
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
            ncx2der2 = -8*s4/(vv**2) - 4*s2*numer/(vv**2) + 1/vv - 4*s2*numer/(vv**2) + 16*s4*(numer)**2/(vv**3)
            der2 = 2*ncx2der + ncx2der2 * (2*mu)**2
            return val,der,der2
        return ff



    ncxloss_gauss(5.5,10,1.0)(2.3),ncxloss_gauss(2.3,4,.1)(.2)




    ((3.9419589177449339, 0.7971603621318025),
     (1024.6527952623899, -7211.666666666661))




    ncxloss_gauss(5.5,10,1.0)(2.3),ncxloss_gauss(2.3,4,.1)(.2)




    ((3.9419589177449339, 0.7971603621318025),
     (1024.6527952623899, -684536.6666666659))




    
    ncxloss_gauss(2.3,4,.1)(.2)


    


    # test derivatives
    df = npr.randint(5,10)
    x = npr.uniform(70,100)
    mus = np.arange(30,70,1.0)
    ncs = mus**2
    sigma = 1.0 #npr.uniform(.5,2.0)
    f = ncxloss_gauss(x,df,sigma)
    def fval(x):
        return f(x)[0]
    zip(f(mus)[2],nc.numderiv2(fval,mus,1e-4))
    0




    0




    


    


    


    


    
