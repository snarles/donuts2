
# Comparing the model fits, NNLS vs gaussian prox


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


Generate the data


    # setting up measurement vectors and fitting vectors
    res_bv = 2
    true_kappa = 1.0
    kappa = 1.0
    bvecs = np.sqrt(kappa) * du.geosphere(res_bv)
    bvecs0 = np.sqrt(true_kappa) * du.geosphere(res_bv)
    n = np.shape(bvecs)[0]
    print("Number of measurement directions is "+str(n))
    sgrid = du.geosphere(10)
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
    
    amat0 = du.ste_tan_kappa(sgrid,bvecs0)
    amat = du.ste_tan_kappa(sgrid,bvecs)

    Number of measurement directions is 42
    Number of candidate directions is 511



    true_k = 3
    # randomly generate parameters
    if randoutside:
        bt0 = np.zeros(pp)
        bt0[npr.randint(0,pp-1,true_k)]=1.0/true_k
        if plotsignal:
            plotb(bt0)


    
    
    total_sigma = 0.1
    df = 10
    true_sigma = total_sigma/np.sqrt(df)
    est_sigma = true_sigma
    
    nreps = 2
    mses_cnnls = np.zeros(nreps); emds_cnnls = np.zeros(nreps);
    mses_fnnls = np.zeros(nreps); emds_fnnls = np.zeros(nreps);
    mses_gauss = np.zeros(nreps); emds_gauss = np.zeros(nreps)
    
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
            
        # NNLS after removing noise floor
        yflo = y - est_sigma**2*df; yflo[yflo < 0] = 0
        bt_fnnls = np.squeeze(spo.nnls(amat,np.squeeze(np.sqrt(yflo)))[0])
        mu_fnnls = np.dot(amat,bt_fnnls)
        yh_fnnls = du.column(mu_fnnls**2) + est_sigma**2*df
        mse_fnnls = nla.norm(np.squeeze(y0) - np.squeeze(yh_fnnls))**2
        if plotsignal:
            print mse_fnnls, loss_emd(bt_fnnls)
    
        # get the nonlinear Gaussian approximation
    
        ls_gauss = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
        lk_gauss = likelihood_func(ls_gauss,amat)
        bt_gauss =tncsolve(amat,ls_gauss,np.array(bt_fnnls),0.0,1e-5)
        mu_gauss = np.dot(amat,bt_gauss)
        yh_gauss = mu_gauss**2 + c0
        mse_gauss =  nla.norm(np.squeeze(y0) - np.squeeze(yh_gauss))**2
        if plotsignal:
            print loss_mse(bt_gauss), loss_emd(bt_gauss), lk_gauss(bt_gauss), sum(bt_gauss > 0)
            
        # record results
        mses_cnnls[iii] = mse_cnnls
        emds_cnnls[iii] = loss_emd(bt_cnnls)
        mses_fnnls[iii] = mse_fnnls
        emds_fnnls[iii] = loss_emd(bt_fnnls)
        mses_gauss[iii] = loss_mse(bt_gauss)
        emds_gauss[iii] = loss_emd(bt_gauss)
    
    (np.mean(mses_cnnls),np.mean(emds_cnnls)),(np.mean(mses_fnnls),np.mean(emds_fnnls)),(np.mean(mses_gauss),np.mean(emds_gauss)) 




    ((0.025063364562029605, 0.31675681471824646),
     (0.02340516841252516, 0.31309619545936584),
     (0.026046474153000876, 0.31247252225875854))



((0.00025949582037454683, 0.045025046421214937),
 (0.00024499683395171512, 0.032382597802206876),
 (0.0002450328055529921, 0.032384660330135373))


    loss_emd(bt_fnnls)




    0.3021826446056366




    bt2 = bt_fnnls
    bt2[0:550] = 1e-3
    sum(bt2 > 0),loss_emd(bt2)

## Selecting sigma


    total_sigma = 0.1
    df = 10
    true_sigma = total_sigma/np.sqrt(df)
    est_sigmas = true_sigma * np.array([0.0,1.0,1.1])
    nsigs = len(est_sigmas)
    
    nreps = 100
    mses_cnnls = np.zeros((nreps,nsigs)); emds_cnnls = np.zeros((nreps,nsigs));
    mses_fnnls = np.zeros((nreps,nsigs)); emds_fnnls = np.zeros((nreps,nsigs));
    mses_gauss = np.zeros((nreps,nsigs)); emds_gauss = np.zeros((nreps,nsigs))
    
    for iii in range(nreps):
        c0 = df*true_sigma**2
        mu = np.dot(amat0,bt0)
        y0 = mu**2 + true_sigma**2*df
        y = ncx.rvs_ncx2(df,mu,0,true_sigma)
        def loss_emd(x0):
            return du.arc_emd(sgrid,ncx.column(bt0),sgrid,ncx.column(x0))
        def loss_mse(x0):
            yh = np.squeeze(np.dot(amat,ncx.column(x0))**2) + true_sigma**2 * df
            return nla.norm(np.squeeze(y0)-np.squeeze(yh))**2
        if not randoutside:
            bt0 = np.zeros(pp)
            bt0[npr.randint(0,pp-1,true_k)]=1.0
        for jj in range(len(est_sigmas)):
            est_sigma = est_sigmas[jj]
    
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
            
            # NNLS after removing noise floor
            yflo = y - est_sigma**2*df; yflo[yflo < 0] = 0
            bt_fnnls = np.squeeze(spo.nnls(amat,np.squeeze(np.sqrt(yflo)))[0])
            mu_fnnls = np.dot(amat,bt_fnnls)
            yh_fnnls = du.column(mu_fnnls**2) + est_sigma**2*df
            mse_fnnls = nla.norm(np.squeeze(y0) - np.squeeze(yh_fnnls))**2
            
            # get the nonlinear Gaussian approximation
            ls_gauss = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
            lk_gauss = likelihood_func(ls_gauss,amat)
            bt_gauss =tncsolve(amat,ls_gauss,np.array(bt_fnnls),0.0,1e-5)
            mu_gauss = np.dot(amat,bt_gauss)
            yh_gauss = mu_gauss**2 + c0
            mse_gauss =  nla.norm(np.squeeze(y0) - np.squeeze(yh_gauss))**2
            
            # record results
            mses_cnnls[iii,jj] = mse_cnnls
            emds_cnnls[iii,jj] = loss_emd(bt_cnnls)
            mses_fnnls[iii,jj] = mse_fnnls
            emds_fnnls[iii,jj] = loss_emd(bt_fnnls)
            mses_gauss[iii,jj] = loss_mse(bt_gauss)
            emds_gauss[iii,jj] = loss_emd(bt_gauss)



    mses_cnnls.mean(axis=0),mses_fnnls.mean(axis=0),mses_gauss.mean(axis=0)




    (array([ 0.01858098,  0.01858098,  0.01858098]),
     array([ 0.01864885,  0.01859295,  0.01858643]),
     array([ 0.02114164,  0.0190149 ,  0.01967004]))




    mses_fnnls.mean(axis=0)




    array([ 0.01864885,  0.01859295,  0.01858643])



## Selecting kappa


    # setting up measurement vectors and fitting vectors
    res_bv = 2
    true_kappa = 2.0
    kappas = true_kappa * np.array([0.8,0.9,1.0,1.1,1.2])
    bvecss = [np.sqrt(kappa) * du.geosphere(res_bv) for kappa in kappas]
    len_kappas = len(kappas)
    bvecs0 = np.sqrt(true_kappa) * du.geosphere(res_bv)
    n = np.shape(bvecs)[0]
    print("Number of measurement directions is "+str(n))
    sgrid = du.geosphere(5)
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
    
    amat0 = du.ste_tan_kappa(sgrid,bvecs0)
    amats = [du.ste_tan_kappa(sgrid,bvecs) for bvecs in bvecss]

    Number of measurement directions is 42
    Number of candidate directions is 126



    true_k = 1
    # randomly generate parameters
    if randoutside:
        bt0 = np.zeros(pp)
        bt0[npr.randint(0,pp-1,true_k)]=1.0/true_k
        if plotsignal:
            plotb(bt0)


    total_sigma = 0.1
    df = 30
    true_sigma = total_sigma/np.sqrt(df)
    est_sigma = true_sigma
    
    nreps = 100
    mses_cnnls = np.zeros((nreps,len_kappas)); emds_cnnls = np.zeros((nreps,len_kappas));
    mses_fnnls = np.zeros((nreps,len_kappas)); emds_fnnls = np.zeros((nreps,len_kappas));
    mses_gauss = np.zeros((nreps,len_kappas)); emds_gauss = np.zeros((nreps,len_kappas));
    
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
        for jj in range(len_kappas):
            amat = amats[jj]
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
    
            # NNLS after removing noise floor
            yflo = y - est_sigma**2*df; yflo[yflo < 0] = 0
            bt_fnnls = np.squeeze(spo.nnls(amat,np.squeeze(np.sqrt(yflo)))[0])
            mu_fnnls = np.dot(amat,bt_fnnls)
            yh_fnnls = du.column(mu_fnnls**2) + est_sigma**2*df
            mse_fnnls = nla.norm(np.squeeze(y0) - np.squeeze(yh_fnnls))**2
            if plotsignal:
                print mse_fnnls, loss_emd(bt_fnnls)
    
            # get the nonlinear Gaussian approximation
    
            ls_gauss = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
            lk_gauss = likelihood_func(ls_gauss,amat)
            bt_gauss =tncsolve(amat,ls_gauss,np.array(bt_fnnls),0.0,1e-5)
            mu_gauss = np.dot(amat,bt_gauss)
            yh_gauss = mu_gauss**2 + c0
            mse_gauss =  nla.norm(np.squeeze(y0) - np.squeeze(yh_gauss))**2
            if plotsignal:
                print loss_mse(bt_gauss), loss_emd(bt_gauss), lk_gauss(bt_gauss), sum(bt_gauss > 0)
    
            # record results
            mses_cnnls[iii,jj] = mse_cnnls
            emds_cnnls[iii,jj] = loss_emd(bt_cnnls)
            mses_fnnls[iii,jj] = mse_fnnls
            emds_fnnls[iii,jj] = loss_emd(bt_fnnls)
            mses_gauss[iii,jj] = loss_mse(bt_gauss)
            emds_gauss[iii,jj] = loss_emd(bt_gauss)
    
    mses_cnnls.mean(axis=0),mses_fnnls.mean(axis=0),mses_gauss.mean(axis=0)




    (array([ 0.11543589,  0.02224897,  0.00429058,  0.00585151,  0.00746194]),
     array([ 0.1447518 ,  0.03539361,  0.00262939,  0.00512049,  0.00659696]),
     array([ 0.12733025,  0.03244735,  0.00272766,  0.00520625,  0.00666189]))




    
