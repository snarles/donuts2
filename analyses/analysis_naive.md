
## Analysis of real data: comparison of signal prediction NNLS vs Guassian prox


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


    def tncsolve(amat,ls,x0,ft = 1e-3): # use TNC to solve
        def f(x0):
            yh = np.dot(amat,x0)
            return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
        def fprime(x0):
            yh = np.dot(amat,x0)
            rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
            return np.dot(rawg.T,amat)
        bounds = [(0.0,100.0)] * len(x0)
        res = spo.fmin_tnc(f,np.squeeze(x0),fprime=fprime,bounds=bounds,ftol=ft)
        return res[0]


    s0 = s0s[5,:]
    ns0 = len(s0)
    sigma = 0.1
    ls = [ncx.ncxloss_gauss(s,df,sigma) for s in s0]
    def lk(mu):
        
    am = np.ones((ns0,1))
    tncsolve()


    df = 32
    file0 = 'roi1_b1000_1'
    bvecs = np.loadtxt('donuts/data/'+file0+'_bvecs.csv',delimiter=',')
    n = np.shape(bvecs)[0]
    data = np.loadtxt('donuts/data/'+file0+'_data.csv',delimiter=',')
    nvox = np.shape(data)[0]
    
    coords = np.array(data[:,2:5],dtype='int32')
    s0s = data[:,5:15]
    ys = data[:,15:]
    print np.shape(coords),np.shape(s0s),np.shape(ys)
    map3d = dict()
    for ii in range(nvox):
        map3d[tuple(coords[ii,])] = ii

    (8000, 3) (8000, 10) (8000, 150)



    def plotz(z=np.ones(n)):
        fig = plt.figure()
        zz = np.ones(n)
        ax = fig.gca(projection='3d')
        ax.scatter(zz*bvecs[:,0],zz*bvecs[:,1],zz*bvecs[:,2])
        plt.show()


    #plotz()
    plotz(data[200,15:])


    sgrid = du.geosphere(8)
    sgrid = sgrid[sgrid[:,2] >= 0,:]
    
    def rand_est_vox(ii,amat,sigma):
        inds = npr.permutation(150)
        inds_te = inds[:10]
        inds_tr = inds[10:]
        return est_vox(ii, inds_te,inds_tr, amat, sigma)
    
    def est_vox(ii, inds_te,inds_tr, amat, sigma):
        n_tr = sum(inds_tr)
        y = data[200,15:]
        y_tr = y[inds_tr]
        y_te = y[inds_te]
        amat_tr = amat[inds_tr,:]
        amat_te = amat[inds_te,:]
        ls = [ncx.ncxloss_gauss(yy,df,sigma) for yy in y_tr]
        bt_nnls = spo.nnls(amat_tr,np.sqrt(y_tr))[0]
        mu_tr_nnls = np.squeeze(np.dot(amat_tr,bt_nnls))
        sigma_nnls = nla.norm(np.sqrt(y_tr) - mu_tr_nnls)**2/n_tr
        mu_nnls = np.squeeze(np.dot(amat_te,bt_nnls))
        yh_nnls = mu_nnls**2 + sigma_nnls**2
        mse_nnls = nla.norm(y_te - yh_nnls)**2
        bt_gp = tncsolve(amat_tr,ls,np.array(bt_nnls))
        mu_gp = np.squeeze(np.dot(amat_te,bt_gp))
        yh_gp = mu_gp**2 + df*sigma**2
        mse_gp = nla.norm(y_te - yh_gp)**2
        return mse_nnls,mse_gp
    
    def search_sigma(ii, inds_te,inds_tr, amat, sigmas):
        res = np.array([est_vox(ii, inds_te,inds_tr, amat, sigma)[1] for sigma in sigmas])
        return res, sigmas[np.where(res==min(res))[0][0]], min(res)


    data[200,15:]




    array([  78.,   18.,   31.,   70.,   35.,   16.,   61.,   25.,   44.,
             74.,   53.,   17.,   55.,   27.,   44.,   52.,   77.,   39.,
             32.,   28.,   18.,   14.,   35.,   29.,   38.,   13.,   25.,
             54.,   47.,   34.,   52.,   45.,   13.,   52.,   52.,   32.,
             15.,   68.,   15.,   29.,   36.,   29.,   19.,   49.,   26.,
             19.,   32.,   13.,   10.,   67.,   12.,   44.,   34.,    6.,
             30.,   43.,   56.,   24.,   38.,   11.,   29.,   19.,   48.,
             28.,   52.,   36.,   66.,   34.,   51.,   26.,   32.,  106.,
             44.,   32.,   52.,   44.,   19.,   89.,   20.,   12.,   26.,
             39.,   55.,   89.,   37.,   21.,   30.,   69.,   52.,   11.,
             45.,   21.,   82.,   16.,   47.,   36.,   23.,   70.,   67.,
             15.,   35.,   31.,   23.,   25.,   15.,   46.,   22.,   48.,
             70.,   28.,   41.,   44.,   14.,   30.,   54.,   25.,   23.,
             70.,   87.,   28.,   20.,   49.,   11.,   21.,   44.,   20.,
             48.,   37.,   58.,   48.,   16.,   32.,   30.,   20.,   46.,
             40.,   19.,   48.,   30.,   42.,   30.,   35.,   60.,   70.,
             50.,   23.,   18.,   54.,   93.,   26.])




    nsub = 100
    vox_sel = npr.permutation(nvox)[:nsub]
    
    
    
    inds = npr.permutation(150)
    inds_te = inds[:10]
    inds_tr = inds[10:]
    n_tr = sum(inds_tr) + 0.0
    
    



    ii=200
    sigma = 0.15
    kappa = 4.0
    amat = du.ste_tan_kappa(sgrid,np.sqrt(kappa)*bvecs)
    print est_vox(200, inds_te,inds_tr, amat, sigma)


    (1356.2071870852833, 1364.5609650260724)



    ii=200
    
    inds = npr.permutation(150)
    inds_te = inds[:10]
    inds_tr = inds[10:]
    n_tr = sum(inds_tr) + 0.0
    
    sigmas = np.arange(0.01,0.2,0.01)
    kappa = 4.0
    amat = du.ste_tan_kappa(sgrid,np.sqrt(kappa)*bvecs)
    print search_sigma(ii, inds_te,inds_tr, amat, sigmas)

    (array([ 3425.30721993,  3425.47309857,  3426.25289273,  3423.81548719,
            3426.68945743,  3424.78663459,  3428.33047049,  3425.97213952,
            3429.35951823,  3428.77387827,  3426.64199916,  3427.31920439,
            3429.96356894,  3428.20794556,  3429.30453633,  3433.04097507,
            3431.71486318,  3431.64674782,  3428.73982805]), 0.040000000000000001, 3423.8154871865981)



    ii=200
    sigmas = np.arange(0.01,0.2,0.01)
    kappa = 3.0
    amat = du.ste_tan_kappa(sgrid,np.sqrt(kappa)*bvecs)
    print search_sigma(ii, inds_te,inds_tr, amat, sigmas)


    (array([ 1554.49471069,  1561.31760164,  1555.25892728,  1560.35320713,
            1557.29641946,  1563.0073963 ,  1560.91609094,  1558.14296856,
            1562.30806514,  1564.19332027,  1572.40297921,  1562.90432936,
            1564.53409288,  1568.08248664,  1561.89984733,  1566.94327951,
            1572.68497665,  1559.8390193 ,  1570.79200607]), 0.01, 1554.4947106895745)



    


    ii=200
    sigmas = np.arange(0.01,0.2,0.01)
    kappa = 4.0
    amat = du.ste_tan_kappa(sgrid,np.sqrt(kappa)*bvecs)
    print search_sigma(ii, inds_te,inds_tr, amat, sigmas)


    sigma = 0.15
    kappa = 4.0
    amat = du.ste_tan_kappa(sgrid,np.sqrt(kappa)*bvecs)
    rand_est_vox(20,amat,sigma)




    (6696.2556603521916, 6822.8819573352685)




    tel = {'kyle' : 40}


    tel[(1,1)] = 5


    tel[(1,1)]




    5




    
