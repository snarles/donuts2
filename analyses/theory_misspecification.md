
## What happens when you choose the wrong $$\kappa$$


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


    bvecs = du.georandsphere(6,20)
    n = np.shape(bvecs)[0]
    
    #rsh = du.rsh_basis(bvecs,6)
    #psh = np.shape(rsh)[1]
    
    perm = npr.permutation(n)
    pbvecs = bvecs[perm[:psh],:]
    xs = du.ste_tan_kappa(2.0*pbvecs,bvecs)
    def plotz(zz=np.ones(n)):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(zz*bvecs[:,0],zz*bvecs[:,1],zz*bvecs[:,2])
        plt.show()


    def histomean(breaks,x,y):
        mdpts = (breaks[1:] + breaks[:-1])/2.0
        k = len(mdpts)
        ans = np.zeros(k)
        for ii in range(k):
            ans[ii] = np.mean(y[np.logical_and(x >= breaks[ii], x < breaks[ii+1])])
        return mdpts,ans


    true_kappa = 3.5
    kappa = 4.0
    xs0 = du.ste_tan_kappa(np.sqrt(true_kappa)*np.eye(3),bvecs)
    xs = du.ste_tan_kappa(np.sqrt(kappa)*bvecs,bvecs)
    y0 = xs0[:,0]
    plotz(y0)


    
    xgrid = np.arange(0,1.01,0.01)
    thetagrid = np.arange(-np.pi,np.pi,np.pi/100.0)
    def g(x):
        return np.exp(-true_kappa * x**2)
    def f(t,x):
        xd = np.tile(du.column(x),(1,len(thetagrid)))
        mxd = np.tile(du.column(np.sqrt(1-x**2)),(1,len(thetagrid)))
        cthetad = np.tile(du.column(np.cos(thetagrid)).T,(len(x),1))
        td = t*np.ones((len(x),len(thetagrid)))
        mtd = np.sqrt(1-t**2)*np.ones((len(x),len(thetagrid)))
        exponent = -kappa*(xd*td + mtd*mxd*cthetad)**2
        return np.exp(exponent).mean(axis=1)

##### Various checks


    % matplotlib inline


    plt.scatter(xgrid,g(xgrid));plt.show()


    plt.scatter(xgrid,f(1.0,xgrid),color="red")
    plt.scatter(xgrid,np.exp(-kappa * xgrid**2))
    plt.show()


    # check the marginals
    plt.scatter(np.absolute(bvecs[:,0]), np.squeeze(y0))
    plt.scatter(xgrid,g(xgrid),color="red")
    plt.show()


    # check the marginals
    ind = 73
    y = xs[:,ind]
    xpar = np.absolute(bvecs[ind,0])
    mdpts, hm = histomean(np.arange(0,1.01,.1),np.absolute(bvecs[:,0]), np.squeeze(y))
    plt.scatter(np.absolute(bvecs[:,0]), np.squeeze(y),color="gray")
    plt.scatter(xgrid,f(xpar,xgrid),color="blue")
    plt.scatter(mdpts,hm,color="red")
    plt.show()

##### Fit!


    # fit!
    ygrid = np.arange(0,1 + 1e-10,0.0001)
    xgrid = np.arange(0,1.001,0.02)
    b = g(ygrid)
    amat = np.vstack([f(x,ygrid) for x in xgrid]).T


    coefs = spo.nnls(amat,b)[0]
    bhat = np.squeeze(np.dot(amat,coefs))
    plt.scatter(xgrid,coefs); plt.show()
    nla.norm(b-bhat),nla.norm(b-bhat,np.inf)




    (4.9963565804689526e-07, 9.8128147274501032e-09)



## Fitting the spherical harmonics model (gaussian case)


    def constrained_ls(y,xs,q,nits=100): #fits y = xs *b, subject to on q*b >= 0, where q is orthogonal
        n = len(y)
        pp = np.shape(xs)[1]
        nc = np.shape(q)[0]
        x0 = np.zeros(n)
        for ii in range(nits):


    res_bv = 4
    kappa = 4.0
    true_kappa = 3.0
    bvecs = np.sqrt(kappa) * du.geosphere(res_bv)
    bvecs0 = np.sqrt(true_kappa) * du.geosphere(res_bv)
    n = np.shape(bvecs)[0]
    print("Number of measurement directions is "+str(n))
    sgrid = du.geosphere(12)
    sgrid = sgrid[sgrid[:,2] >= 0,:]
    pp = np.shape(sgrid)[0]
    print("Number of candidate directions is "+str(pp))
    amat0 = du.ste_tan_kappa(sgrid,bvecs0)
    amat = du.ste_tan_kappa(sgrid,bvecs)
    rsh = du.rsh_basis(sgrid,6)
    q_sh,r = nla.qr(rsh)
    pprsh = np.shape(rsh)[1]
    xs_sh = np.dot(amat,q_sh)
    print("Number of spherical harmonics is "+str(pprsh))

    Number of measurement directions is 162
    Number of candidate directions is 731
    Number of spherical harmonics is 49



    # randomly generate parameters, GAUSSIAN case
    true_k = 3
    
    true_sigma = 0.0
    #bt0 = np.absolute(npr.normal(0,1,pp)) * npr.binomial(1,3.0/pp,pp)
    bt0 = np.zeros(pp)
    bt0[npr.randint(0,pp-1,true_k)]=1.0
    y0 = np.dot(amat0,bt0)
    y = y0 + true_sigma * npr.normal(0,1,n)
    def loss_emd(x0):
        return du.arc_emd(sgrid,ncx.column(bt0),sgrid,ncx.column(x0))
    def loss_mse(x0):
        yh = np.squeeze(np.dot(amat,ncx.column(x0)))
        return nla.norm(np.squeeze(y0)-np.squeeze(yh))**2


    # NNLS solution
    bt_nnls = np.squeeze(spo.nnls(amat,np.squeeze(y))[0])
    yh_nnls = np.dot(amat,bt_nnls)
    print loss_mse(bt_nnls), loss_emd(bt_nnls)

    4.57428391463e-28 0.224979832768



    # UNRESTRICTED rsh solution
    gam_ush = nla.lstsq(xs_sh,y)[0]
    bt_ush = np.dot(q_sh,gam_ush)
    yh_ush = np.dot(xs_sh,gam_ush)
    yh_ush_check = np.dot(amat,bt_ush)


    plt.scatter(bt_ush,bt_nnls);plt.show()


    xs = xs_sh; q = q_sh


    n = len(y)
    pp = np.shape(xs)[1]
    nc = np.shape(q)[0]
    x0 = np.zeros(pp)


    


    yh = np.dot(xs,x0)
    r = y-yh
    hess = np.dot(xs.T,xs)
    grad = np.dot(xs.T,-r)
    newton_dir = nla.lstsq(hess,grad)[0]
    eps = .1


    def loss_f(x0):
        yh = np.dot(xs,x0)
        return nla.norm(y-yh)**2


    x1 = x0 - eps * newton_dir
    ch = np.dot(q,x1)



    sum(ch < 0)




    282




    ch2 = np.array(ch)
    ch2[ch2 < 0] = 0
    x2 = np.dot(q.T,ch2)
    ch2 = np.dot(q,x2)
    sum(ch2 < 0)




    185




    ch2 = np.array(ch2)
    ch2[ch2 < 0] = 0
    x2 = np.dot(q.T,ch2)
    ch2 = np.dot(q,x2)
    sum(ch2 < 0)




    27




    
