
## Testing alternative to NNLS: use of gaussian approximation to non-central chi squared

### 0) Run this before any other section


    import os
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
    
    def ncx2gauss(x,df):
        def ff(nc):
            val= .5*np.log(2*np.pi*(2*df+4*nc)) + (nc+df-x)**2/(4*df+8*nc)
            ncx2der= 1/(df + 2*nc) + 2*(nc + df-x)/(4*df + 8*nc) - 8*((nc+df-x)/(4*df+8*nc))**2
            return val, ncx2der
        return ff
    
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
    
    def ncx2true(x,df):
        val0 = -spst.chi2.logpdf(x,df)
        def ff(nc):
            def calcval(ncc):
                val = np.zeros(len(ncc))
                val[ncc !=0] = -spst.ncx2.logpdf(x,df,ncc[ncc!=0])
                val[ncc ==0] = val0
                return val
            val = calcval(np.atleast_1d(nc))
            dval = calcval(np.atleast_1d(nc + 1e-3))
            ncx2der = 1e3*(dval-val)
            return scalarize(val), scalarize(ncx2der)
        return ff
    
    def mean_ncx(df,nc0,sigma=1.0): # approximate mean
        nc = nc0/sigma
        mu = nc0+df
        mu = df + nc
        sig2 = 2*df + 4*nc
        the_mean = np.sqrt(mu) - sig2/(8* np.power(mu,1.5))
        return the_mean*sigma
    
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
    
    def bfgssolve(amat,ls,x0,lb=0.0,nd = True): # use LBFS-G to solve, lb = lower bound, nd = numerical derivative
        if nd:
            def f(x0):
                yh = np.dot(amat,x0)
                return sum(np.array([ls[i](yh[i]) for i in range(len(yh))]))
            def fprime(x0):
                yh = np.dot(amat,x0)
                rawg= np.array([numderiv(ls[i],yh[i],1e-3) for i in range(len(yh))])
                return np.dot(rawg.T,amat)
        else:
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

### 1) Basic examples

Visual comparison of gaussian loss and true loss function


    x = 100.0
    df = 10
    ub = 4*x
    ncs = np.arange(ub/100,ub,ub/100)
    ll_true = ncx2true(x,df)
    ll_gauss = ncx2gauss(x,df)
    plt.scatter(ncs,ll_true(ncs)[0])
    plt.scatter(ncs,ll_gauss(ncs)[0],color="green")
    plt.scatter(ncs[np.isnan(ll_true(ncs)[0])],0.0*ncs[np.isnan(ll_true(ncs)[0])],color="red")
    plt.show()
    (max(ll_true(ncs)[0]),min(ll_true(ncs)[0]))




    (56.053404410035199, 3.8929547094843855)



Comparison of NNLS and gaussian approximation on sparse recovery (random
vectors)


    n = 10000
    p = 1000
    amat = np.absolute(npr.normal(0,1,(n,p)))
    bt0 = np.zeros(p)
    bt0[:4] = 5
    df = 10
    mu = np.dot(amat,bt0)
    ysq = spst.ncx2.rvs(df,mu**2)
    ls_gauss = [ncxloss_gauss(y,df) for y in ysq]
    ls_mean = [ncxloss_mean(y,df) for y in ysq]
    ls_true = [ncxloss_true(y,df) for y in ysq]
    def recovery_score(x0): # higher score is better
        diff = x0-bt0
        return (-sum(diff[diff > 0]), sum(diff[diff <0]))
    bt = np.squeeze(spo.nnls(amat,np.squeeze(np.sqrt(ysq)))[0])
    res_gauss = bfgssolve(amat,ls_gauss,np.array(bt),0.0,False)
    res_mean = bfgssolve(amat,ls_mean,np.array(bt),0.0,False)
    #res_true = bfgssolve(amat,ls_true,np.array(bt),0.0,False)
    #[recovery_score(bt),recovery_score(res_gauss[0]),recovery_score(res_true[0])]
    [recovery_score(bt),recovery_score(res_gauss[0]),recovery_score(res_mean[0])]




    [(-0.91227300991181082, -0.48514542007171357),
     (-0.1338259227374903, -0.11260504714007435),
     (-0.23376039853958405, -0.15947981901076247)]




    # initialization effect?
    res_gauss = bfgssolve(amat,ls_gauss,np.array(bt0),0.0,False)
    res_mean = bfgssolve(amat,ls_mean,np.array(bt0),0.0,False)
    #res_true = bfgssolve(amat,ls_true,np.array(bt),0.0,False)
    #[recovery_score(bt),recovery_score(res_gauss[0]),recovery_score(res_true[0])]
    [recovery_score(bt),recovery_score(res_gauss[0]),recovery_score(res_mean[0])]




    [(-1.2733907302607161, -0.74638956236892984),
     (-0.52208899451310531, -0.39691630963561231),
     (-0.80155205333020674, -0.52078907593583157)]



Test formula for mean of ncx


    def mean_test(df,nc):
        xsq = spst.ncx2.rvs(df,nc,size=10000)
        emp_mean = np.mean(np.sqrt(xsq))
        mu = df + nc
        sig2 = 2*df + 4*nc
        #the_mean = np.sqrt(mu) - sig2/(8* np.power(mu,1.5))
        the_mean = mean_ncx(df,nc)
        return emp_mean,the_mean


    mean_test(10,100.0)




    (10.429558662907173, 10.442582312669339)



### 2) Testing on diffusion model


    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    os.chdir("..")
    import donuts.deconv.utils as du

Generate the data


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
    plotsignal = False

    Number of measurement directions is 252
    Number of candidate directions is 341



    # randomly generate parameters
    true_k = 2
    true_vs = du.normalize_rows(npr.normal(0,1,(true_k,3)))
    true_vs[:,2] = np.absolute(true_vs[:,2])
    true_ws = 0.5*np.ones((true_k,1))/true_k
    true_sigma = 0.1
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


    amat = du.ste_tan_kappa(sgrid,bvecs)
    df = 10
    ysq = (np.squeeze(y1)/true_sigma)**2
    #ysq = spst.ncx2.rvs(df,mu**2)
    ls_gauss = [ncxloss_gauss(y,df) for y in ysq]
    ls_mean = [ncxloss_mean(y,df) for y in ysq]
    ls_true = [ncxloss_true(y,df) for y in ysq]
    bt = np.squeeze(spo.nnls(amat,np.squeeze(np.sqrt(ysq)))[0])
    res_mean = bfgssolve(amat,ls_mean,np.array(bt/true_sigma),0.0,False)
    bt_mean = res_mean[0]*true_sigma
    #res_gauss = bfgssolve(amat,ls_gauss,np.array(bt/true_sigma),0.0,False)
    #bt_gauss = res_gauss[0]*true_sigma
    #res_true = bfgssolve(amat,ls_true,np.array(bt/true_sigma),0.0,False)
    #bt_true = res_true[0]*true_sigma
    
    
    def loss_emd(x0):
        return du.arc_emd(true_vs,true_ws,sgrid,column(x0))
    loss_emd(bt),loss_emd(bt_mean)#,loss_emd(bt_mean)


    


    
