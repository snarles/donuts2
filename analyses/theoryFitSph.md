
### Fitting Spherical Harmonics


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


    import cvxopt as cvx

    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)



    
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
    
    def slsqpsolve(amat,ls,qmat,x0,ft = 1e-3): # use TNC to solve, lb = lower bound
        def f(x0):
            yh = np.dot(amat,x0)
            return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
        def fprime(x0):
            yh = np.dot(amat,x0)
            rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
            return np.dot(rawg.T,amat)
        ieqcons = [lambda x, q=q: sum(q * x) for q in qmat]
        ieqprime = [lambda q=q: q for q in qmat]
        res = spo.fmin_slsqp(f,np.squeeze(x0),ieqcons=ieqcons,fprime=fprime,fprime_ieqcons = ieqprime,acc=ft)
        return res
    
    def likelihood_func(ls,amat):
        def lk(x0):
            yh = np.dot(amat,np.reshape(x0,(-1,1)))
            return sum([ls[ii](yh[ii])[0] for ii in range(len(ls))])
        return lk



    bvecs = du.georandsphere(4,1)
    n = np.shape(bvecs)[0]
    sgrid0 = du.georandsphere(3,1)
    sgrid0 = sgrid0[sgrid0[:,2] > 0,:]
    sgrid = du.georandsphere(5,2)
    sgrid = sgrid[sgrid[:,2] > 0,:]
    pp0 = np.shape(sgrid0)[0]
    pp = np.shape(sgrid)[0]
    rsh = du.rsh_basis(sgrid,15)
    pp_rsh = np.shape(rsh)[1]
    print((n,pp,pp_rsh))


    (162, 252, 252)



    # setting up measurement vectors and fitting vectors
    true_kappa = 4.0
    kappa = 4.0
    def plotb(zz=np.ones(pp)):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(sgrid[:,0],sgrid[:,1],sgrid[:,2])
        ax.scatter(zz*sgrid[:,0],zz*sgrid[:,1],zz*sgrid[:,2],color="red")
        plt.show()
    # do you want plots?
    plotsignal = True
    # randomize parameters outside of loop?
    randoutside = True
    
    amat0 = du.ste_tan_kappa(sgrid0,np.sqrt(true_kappa) *bvecs)
    amat = du.ste_tan_kappa(sgrid,np.sqrt(kappa) *bvecs)
    amat_rsh = np.dot(amat, rsh)


    true_k = 1
    # randomly generate parameters
    if randoutside:
        bt0 = np.zeros(pp0)
        bt0[npr.randint(0,pp0-1,true_k)]=1.0/true_k


    total_sigma = 0.2
    df = 10
    true_sigma = total_sigma/np.sqrt(df)
    est_sigma = true_sigma
    
    c0 = df*true_sigma**2
    mu = np.dot(amat0,bt0)
    y0 = mu**2 + true_sigma**2*df
    y = ncx.rvs_ncx2(df,mu,0,true_sigma)
    def loss_emd(x0):
        return du.arc_emd(sgrid0,ncx.column(bt0),sgrid,ncx.column(x0))
    def loss_mse(x0):
        yh = np.squeeze(np.dot(amat,ncx.column(x0))**2) + true_sigma**2 * df
        return nla.norm(np.squeeze(y0)-np.squeeze(yh))**2
    
    # NNLS after removing noise floor
    yflo = y - est_sigma**2*df; yflo[yflo < 0] = 0
    bt_fnnls = np.squeeze(spo.nnls(amat,np.squeeze(np.sqrt(yflo)))[0])
    mu_fnnls = np.dot(amat,bt_fnnls)
    yh_fnnls = du.column(mu_fnnls**2) + est_sigma**2*df
    mse_fnnls = nla.norm(np.squeeze(y0) - np.squeeze(yh_fnnls))**2
    if plotsignal:
        print mse_fnnls, loss_emd(bt_fnnls)
        #plotb(bt_fnnls)

    0.127051003454 0.0970079600811



    # use CVXopt to solve CSD
    b = np.squeeze(np.sqrt(yflo))
    pmat = cvx.matrix(2.0*np.dot(amat_rsh.T,amat_rsh))
    qvec = cvx.matrix(-2.0*np.squeeze(np.dot(b.T,amat_rsh)))
    qmat = cvx.matrix(-rsh)
    hvec = cvx.matrix(np.zeros(pp))
    res = cvx.solvers.qp(pmat,qvec,qmat,hvec)
    gamma_csd = np.squeeze(np.array(res['x']))
    bt_csd = np.squeeze(np.dot(rsh,gamma_csd))
    #bt_csd[bt_csd < 1e-3] = 0
    mu_csd = np.dot(amat_rsh,gamma_csd)
    #mu_csd = np.dot(amat,bt_csd)
    yh_csd = du.column(mu_csd**2) + est_sigma**2*df
    mse_csd = nla.norm(np.squeeze(y0) - np.squeeze(yh_csd))**2
    if plotsignal:
        print mse_csd, loss_emd(bt_csd)

         pcost       dcost       gap    pres   dres
     0: -4.9717e+01 -5.0655e+01  3e+02  2e+01  2e-02
     1: -4.9656e+01 -5.0496e+01  9e+00  4e-01  5e-04
     2: -4.9537e+01 -4.9590e+01  1e+00  8e-02  8e-05
     3: -4.9478e+01 -4.9554e+01  1e+00  4e-02  4e-05
     4: -4.9439e+01 -4.9472e+01  2e-01  9e-03  9e-06
     5: -4.9417e+01 -4.9447e+01  3e-02  7e-16  1e-16
     6: -4.9435e+01 -4.9443e+01  9e-03  5e-16  3e-16
     7: -4.9441e+01 -4.9442e+01  8e-04  6e-16  1e-16
     8: -4.9442e+01 -4.9442e+01  3e-05  5e-16  5e-16
    Optimal solution found.
    0.127101188515 0.0970028266311



    # get the nonlinear Gaussian approximation
    
    ls_gauss = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
    lk_gauss = likelihood_func(ls_gauss,amat)
    bt_gauss =tncsolve(amat,ls_gauss,np.array(bt_fnnls),0.0,1e-6)
    mu_gauss = np.dot(amat,bt_gauss)
    yh_gauss = mu_gauss**2 + c0
    mse_gauss =  nla.norm(np.squeeze(y0) - np.squeeze(yh_gauss))**2
    if plotsignal:
        print loss_mse(bt_gauss), loss_emd(bt_gauss), lk_gauss(bt_gauss), sum(bt_gauss > 0)

    0.0647017839578 0.0359607078135 [-250.38031699] 5



    # use slsqp to get the CSD
    qmat = rsh
    x0 = np.array(gamma_csd)
    ft = 1e-3
    ls = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
    gamma_gcsd = slsqpsolve(amat_rsh,ls,qmat,x0,ft = 1e-4)
    bt_gcsd = np.squeeze(np.dot(rsh,gamma_gcsd))
    if plotsignal:
        print loss_mse(bt_gcsd), loss_emd(bt_gcsd), lk_gauss(bt_gcsd), sum(bt_gcsd > 0), sum(bt_gcsd < 0)

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -250.380334531
                Iterations: 10
                Function evaluations: 27
                Gradient evaluations: 10
    0.0646978271824 0.0359539687634 [-250.38033453] 33 48



    %timeit slsqpsolve(amat_rsh,ls,qmat,x0,ft = 1e-4)

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -191.25889512
                Iterations: 27
                Function evaluations: 76
                Gradient evaluations: 27
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -191.25889512
                Iterations: 27
                Function evaluations: 76
                Gradient evaluations: 27
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -191.25889512
                Iterations: 27
                Function evaluations: 76
                Gradient evaluations: 27
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -191.25889512
                Iterations: 27
                Function evaluations: 76
                Gradient evaluations: 27
    1 loops, best of 3: 9.85 s per loop



    %timeit tncsolve(amat,ls_gauss,np.array(bt_fnnls),0.0,1e-5)

    1 loops, best of 3: 3.73 s per loop


##### try a new basis


    np.shape(r)




    (162, 181)




    plt.scatter(np.array(range(n)),np.cumsum(np.array([nla.norm(rr) for rr in r])))
    plt.show()


    q,r = nla.qr(amat.T)
    qs = q[:,range(150)]
    amat_q = np.dot(amat,qs)
    qmat = qs
    b = np.squeeze(np.sqrt(yflo))
    pmat = cvx.matrix(2.0*np.dot(amat_q.T,amat_q))
    qvec = cvx.matrix(-2.0*np.squeeze(np.dot(b.T,amat_q)))
    qmat = cvx.matrix(-qs)
    hvec = cvx.matrix(np.zeros(pp))
    res = cvx.solvers.qp(pmat,qvec,qmat,hvec)
    gamma_qs = np.squeeze(np.array(res['x']))
    bt_qs = np.squeeze(np.dot(qs,gamma_qs))
    #bt_csd[bt_csd < 1e-3] = 0
    mu_qs = np.dot(amat_q,gamma_qs)
    #mu_csd = np.dot(amat,bt_csd)
    yh_qs = du.column(mu_qs**2) + est_sigma**2*df
    mse_qs = nla.norm(np.squeeze(y0) - np.squeeze(yh_qs))**2
    if plotsignal:
        print mse_qs, loss_emd(bt_qs)

         pcost       dcost       gap    pres   dres
     0: -5.0595e+01 -5.1526e+01  2e+02  1e+01  2e-02
     1: -5.0507e+01 -5.1280e+01  8e+00  5e-01  6e-04
     2: -5.0372e+01 -5.0368e+01  1e+00  8e-02  1e-04
     3: -5.0146e+01 -5.0023e+01  5e-01  3e-02  4e-05
     4: -4.9984e+01 -4.9866e+01  3e-01  1e-02  2e-05
     5: -4.9850e+01 -4.9784e+01  1e-01  5e-03  7e-06
     6: -4.9751e+01 -4.9744e+01  5e-02  1e-03  1e-06
     7: -4.9734e+01 -4.9738e+01  2e-02  3e-04  4e-07
     8: -4.9732e+01 -4.9734e+01  4e-03  4e-05  5e-08
     9: -4.9733e+01 -4.9734e+01  5e-04  4e-06  5e-09
    10: -4.9733e+01 -4.9733e+01  7e-06  5e-08  6e-11
    Optimal solution found.
    0.668907870554 0.245599776506



    
    
    ft = 1e-3
    ls = [ncx.ncxloss_gauss(yy,df,est_sigma) for yy in y]
    gamma_gcsd = slsqpsolve(amat_rsh,ls,qmat,x0,ft = 1e-4)
    bt_gcsd = np.squeeze(np.dot(rsh,gamma_gcsd))
    if plotsignal:
        print loss_mse(bt_gcsd), loss_emd(bt_gcsd), lk_gauss(bt_gcsd), sum(bt_gcsd > 0), sum(bt_gcsd < 0)
