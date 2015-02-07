
# Computing the realistic noise model fits


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


    
    def tncsolve(amat,ls,x0,lb=0.0,ft = 1e-3): # use TNC to solve, lb = lower bound
        def f(x0):
            yh = np.dot(amat,x0)
            return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
        def fprime(x0):
            yh = np.dot(amat,x0)
            rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
            return np.dot(rawg.T,amat)
        bounds = [(lb,100.0)] * len(x0)
        res = spo.fmin_tnc(f,np.squeeze(x0),fprime=fprime,bounds=bounds,ftol=ft)
        return res[0]
    
    def likelihood_func(ls,amat):
        def lk(x0):
            yh = np.dot(amat,np.reshape(x0,(-1,1)))
            return sum([ls[ii](yh[ii])[0] for ii in range(len(ls))])
        return lk



    def gd_solve_it(amat,ls,x0,eps): # iteration of gradient descent
        yh = np.squeeze(np.dot(amat,x0))
        def lks(mu):
            res = [ls[ii](mu[ii]) for ii in range(len(ls))]
            vals = np.array([v[0] for v in res])
            lk0 = sum(vals)
            ds = np.array([v[1] for v in res])
            d2s = np.array([v[2] for v in res])
            return lk0,ds,d2s
        lk0,ds,d2s = lks(yh)
        grad = np.dot(amat.T, ds)
        x1 = x0 - eps*grad
        x1[x1 < 0] = 0
        return x1
    
    def gd_activeset(amat,ls,x0,eps,nits=10,asnits=5):
        x0 = np.array(bt_cnnls) + 0.01 * npr.normal(0,1,pp)
        for ii in range(nits):
            x1 = gd_solve_it(amat,ls_gauss,x0,eps/(ii+1))
            s = (x1 > 0)
            x1s = x1[s]
            for jj in range(asnits):
                #x2s = nnnr_solve_it(amat[:,s],ls_gauss,x1s,eps/(ii+1))
                x2s = tncsolve(amat[:,s],ls_gauss,x1s)
                x1s = np.array(x2s)
            x1[s] = x2s
            #print lk_gauss(x1), sum(x1 > 0), loss_mse(x1),loss_emd(x1)
            x0 = np.array(x1)
        return x1


    def s_nnnr_solve_it(amat,ls,x0,eps): # iteration of non-negative Newton-Raphson (Stochastic hessian)
        pp = np.shape(amat)[1]
        yh = np.squeeze(np.dot(amat,x0))
        def lks(mu):
            res = [ls[ii](mu[ii]) for ii in range(len(ls))]
            vals = np.array([v[0] for v in res])
            lk0 = sum(vals)
            ds = np.array([v[1] for v in res])
            d2s = np.array([v[2] for v in res])
            return lk0,ds,d2s
        lk0,ds,d2s = lks(yh)
        inds = npr.permutation(n)[:100]
        grad = np.dot(amat[inds,:].T, ds[inds])
        hess = np.dot(np.dot(amat[inds,:].T, np.diag(d2s[inds])),amat[inds,:])
        mod_hess = eps*hess + np.eye(pp)
        mod_grad = eps*grad - np.dot(mod_hess,x0)
        x1 = spo.nnls(mod_hess,-mod_grad)[0]
        return x1


    def nnnr_lazy_it(amat,ls,x0,eps,nits=10,hnits = 5): # updates the hessian lazily
        pp = np.shape(amat)[1]
        yh = np.squeeze(np.dot(amat,x0))
        def lks(mu):
            res = [ls[ii](mu[ii]) for ii in range(len(ls))]
            vals = np.array([v[0] for v in res])
            lk0 = sum(vals)
            ds = np.array([v[1] for v in res])
            d2s = np.array([v[2] for v in res])
            return lk0,ds,d2s
        lk0,ds,d2s = lks(yh)
        hess = np.dot(np.dot(amat.T, np.diag(d2s)),amat)
        for j in range(hnits):
            grad = np.dot(amat.T, ds)
            mod_hess = eps*hess + np.eye(pp)
            mod_grad = eps*grad - np.dot(mod_hess,x0)
            x0 = spo.nnls(mod_hess,-mod_grad)[0]
        return x0


    def nnnr_solve_it(amat,ls,x0,eps): # iteration of non-negative Newton-Raphson
        pp = np.shape(amat)[1]
        yh = np.squeeze(np.dot(amat,x0))
        def lks(mu):
            res = [ls[ii](mu[ii]) for ii in range(len(ls))]
            vals = np.array([v[0] for v in res])
            lk0 = sum(vals)
            ds = np.array([v[1] for v in res])
            d2s = np.array([v[2] for v in res])
            return lk0,ds,d2s
        lk0,ds,d2s = lks(yh)
        grad = np.dot(amat.T, ds)
        hess = np.dot(np.dot(amat.T, np.diag(d2s)),amat)
        mod_hess = eps*hess + np.eye(pp)
        mod_grad = eps*grad - np.dot(mod_hess,x0)
        x1 = spo.nnls(mod_hess,-mod_grad)[0]
        return x1
    
    def nnnr_activeset(amat,ls,x0,eps,nits=10,asnits=5):
        x0 = np.array(bt_cnnls) + 0.01 * npr.normal(0,1,pp)
        for ii in range(nits):
            x1 = nnnr_solve_it(amat,ls_gauss,x0,eps)
            s = (x1 > 0)
            x1s = x1[s]
            for jj in range(asnits):
                #x2s = nnnr_solve_it(amat[:,s],ls_gauss,x1s,eps/(ii+1))
                x2s = tncsolve(amat[:,s],ls_gauss,x1s,1e-3)
                x1s = np.array(x2s)
            x1[s] = x2s
            #print lk_gauss(x1), sum(x1 > 0), loss_mse(x1),loss_emd(x1)
            x0 = np.array(x1)
        return x1

Generate the data


    


    # setting up measurement vectors and fitting vectors
    res_bv = 3
    true_kappa = 3.0
    kappa = 3.0
    bvecs = np.sqrt(kappa) * du.geosphere(res_bv)
    bvecs0 = np.sqrt(true_kappa) * du.geosphere(res_bv)
    n = np.shape(bvecs)[0]
    print("Number of measurement directions is "+str(n))
    sgrid = du.geosphere(12)
    sgrid = sgrid[sgrid[:,2] >= 0,:]
    pp = np.shape(sgrid)[0]
    print("Number of candidate directions is "+str(pp))
    def plotb(zz=np.ones(pp)):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(sgrid[:,0],sgrid[:,1],sgrid[:,2])
        ax.scatter(zz*sgrid[:,0],zz*sgrid[:,1],zz*sgrid[:,2],color="red")
        plt.show()
    
    # do you want plots?
    plotsignal = False
    # randomize parameters outside of loop?
    randoutside = True

    Number of measurement directions is 92
    Number of candidate directions is 731



    # randomly generate parameters
    true_k = 3
    amat0 = du.ste_tan_kappa(sgrid,bvecs0)
    amat = du.ste_tan_kappa(sgrid,bvecs)
    #bt0 = np.absolute(npr.normal(0,1,pp)) * npr.binomial(1,3.0/pp,pp)
    nreps = 100
    mses_cnnls = np.zeros(nreps); emds_cnnls = np.zeros(nreps); mses_gauss = np.zeros(nreps); emds_gauss = np.zeros(nreps)
    if randoutside:
        bt0 = np.zeros(pp)
        bt0[npr.randint(0,pp-1,true_k)]=1.0
        if plotsignal:
            plotb(bt0)


    
    
    total_sigma = 0.2
    df = 2
    true_sigma = total_sigma/np.sqrt(df)
    est_sigma = true_sigma
    
    for iii in range(nreps):
        if not randoutside:
            bt0 = np.zeros(pp)
            bt0[npr.randint(0,pp-1,true_k)]=1.0
    
        c0 = df*true_sigma**2
        mu = np.dot(amat0,bt0)
        y0 = mu**2 + true_sigma**2*df
        y = ncx.rvs_ncx2(df,mu,0,true_sigma)
        def loss_emd(x0):
            return du.arc_emd(sgrid,ncx.column(bt0),sgrid,ncx.column(x0))
        def loss_mse(x0):
            yh = np.squeeze(np.dot(amat,ncx.column(x0))**2) + true_sigma**2 * df
            return nla.norm(np.squeeze(y0)-np.squeeze(yh))**2
        #y = y0
        # plot the noiseless params
        if plotsignal:
            plotb(bt0)
    
        # penalized NNLS solution with constant term
        yp = np.hstack([np.squeeze(y),0.0])
        l1p = 0.0
        amatc = np.hstack([np.ones((n+1,1)),np.vstack([amat,l1p * np.ones((1,pp))])])
        cbt_cnnls = np.squeeze(spo.nnls(amatc,np.squeeze(np.sqrt(yp)))[0])
        bt_cnnls = cbt_cnnls[1:]
        c_cnnls = cbt_cnnls[0]
        mu_cnnls = np.dot(amat,bt_cnnls)+c_cnnls
        sigma2_cnnls = nla.norm(du.column(np.sqrt(y)) - du.column(mu_cnnls))**2/n
        yh_cnnls = du.column(mu_cnnls**2) + sigma2_cnnls
        mse_cnnls = nla.norm(np.squeeze(y0) - np.squeeze(yh_cnnls))**2
        if plotsignal:
            print mse_cnnls, loss_emd(bt_cnnls), c_cnnls
    
        # get the nonlinear Gaussian approximation
    
        ls_gauss = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
        lk_gauss = likelihood_func(ls_gauss,amat)
        bt_gauss =tncsolve(amat,ls_gauss,np.array(bt_cnnls),0.0,1e-5)
        mu_gauss = np.dot(amat,bt_gauss)
        yh_gauss = mu_gauss**2 + c0
        mse_gauss =  nla.norm(np.squeeze(y0) - np.squeeze(yh_gauss))**2
        if plotsignal:
            print loss_mse(bt_gauss), loss_emd(bt_gauss), lk_gauss(bt_gauss), sum(bt_gauss > 0)
            
        # record results
        mses_cnnls[iii] = mse_cnnls
        emds_cnnls[iii] = loss_emd(bt_cnnls)
        mses_gauss[iii] = loss_mse(bt_gauss)
        emds_gauss[iii] = loss_emd(bt_gauss)
    
    (np.mean(mses_cnnls),np.mean(emds_cnnls)),(np.mean(mses_gauss),np.mean(emds_gauss)) 




    ((2.4973714305063912, 0.14538156069815159),
     (2.5113318215147888, 0.13841385368257761))




    plotb(1.1*bt0)


    

    1.12076800424 0.20382630825 [ 19.71330841] 25



    % timeit tncsolve(amat,ls_gauss,np.array(bt_cnnls),0.0,1e-5)

    1 loops, best of 3: 20.8 s per loop



    # use nnnr active set
    bt_nnnr = nnnr_activeset(amat,ls_gauss,np.array(bt_cnnls),1e-8,nits=10,asnits=1)
    print loss_mse(bt_nnnr), loss_emd(bt_nnnr),lk_gauss(bt_nnnr), sum(bt_nnnr > 0)

    1.14283597235 0.195285573602 [ 25.56100362] 485



    % timeit nnnr_activeset(amat,ls_gauss,np.array(bt_cnnls),1e-8,nits=6,asnits=1)

    1 loops, best of 3: 3.93 s per loop


## Plots of the solutions


    plt.scatter(y0,yh_nnls)
    plt.scatter(y0,yh_nngn,color="red")
    #plt.scatter(y0,yh_pnnls,color="purple")
    #plt.scatter(y0,yh_pnngn,color="orange")
    plt.show()


    print (sigma2_nnls,true_sigma**2 * df)
    print (c_nngn, true_sigma**2 * df)

    (9.2836852703981321e-05, 0.001)
    (0.0, 0.001)



    bt = np.squeeze(spo.nnls(amat,np.squeeze(mu))[0])
    muh = np.dot(amat,bt)
    plt.scatter(mu,muh); plt.show()


    #res_true =tncsolve(amat,ls_true,np.array(bt_gauss/true_sigma),0.0,False)
    #bt_true= res_true[0]*true_sigma


    def loss_emd(x0):
        return du.arc_emd(sgrid,ncx.column(bt0),sgrid,ncx.column(x0))
    def loss_mse(x0):
        yh = np.squeeze(np.dot(amat,ncx.column(x0))**2) + sigma**2 * df
        return nla.norm(np.squeeze(y0)-np.squeeze(yh))**2
    #def loss_mse(x0):
    #    yh = np.squeeze(np.dot(amat,ncx.column(x0)))
    #    return nla.norm(yh-mu)**2/n
    loss_emd(bt0),loss_emd(bt),loss_emd(bt_gauss),loss_emd(bt_gauss_cht)#,loss_emd(bt_true)
    loss_mse(bt0),nla.norm(yhsq_nnls-du.column(ysq))**2,loss_mse(bt_gauss),loss_mse(bt_gauss_cht)


    


    lk_true(bt0),lk_true(bt_gauss)


    lk_gauss(bt0),lk_gauss(bt),lk_gauss(bt_gauss),lk_gauss(bt_gauss_cht)


    np.mean(ysq),np.mean(musq)+df


    zip(np.dot(amat,bt0),mu)
    yh = np.dot(amat,ncx.column(bt_gauss))
    #zip(np.dot(amat,bt0),mu,yh)


    


    # developing Gauss-newton
    def gauss_newton(amat,x0,df,sigma):
        
