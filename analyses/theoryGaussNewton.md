
## Solving $$\min ||y - (Ax)^2||^2$$


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


    # generate the data, a linear model
    n = 100
    p = 10
    sigma = 0.1
    amat = npr.normal(0,1,(n,p))
    bt0 = npr.normal(0,1,p)
    mu0 = np.squeeze(np.dot(amat,bt0))
    y = mu0**2 + sigma*npr.normal(0,1,n)


    # solve using scipy
    def f(bt):
        yh = np.squeeze(np.dot(amat,bt))
        return y - yh**2
    
    res = spo.leastsq(f,npr.normal(0,1,p))
    res[0],bt0




    (array([-0.13989339, -0.37095447,  1.07290576, -2.0152234 , -0.16944098,
             0.37050537, -0.43821162,  0.50825034,  0.07959911, -1.72941008]),
     array([-0.14055573, -0.37405343,  1.07335199, -2.01524983, -0.16841217,
             0.37186173, -0.4421378 ,  0.50955644,  0.07961602, -1.7293152 ]))




    # define our own function
    def gauss_newton_it(y,amat,x0):
        yh = np.squeeze(np.dot(amat,x0))
        r = y-yh**2
        j = 2*np.dot(np.diag(yh),amat)
        dx = nla.lstsq(j,r)[0]
        x1 = x0 + dx
        return x1
    
    def gauss_newton(y,amat,x0,nits):
        for ii in range(nits):
            x0 = gauss_newton_it(y,amat,np.array(x0))
        return x0
    
    gauss_newton(y,amat,npr.normal(0,1,p),100),bt0




    (array([-0.19270869, -0.07899578,  0.96046007, -0.42387723,  1.3141612 ,
             3.02313946, -0.72212011, -1.35554165, -1.35691121, -1.36725985]),
     array([-0.19600593, -0.06909837,  0.94315796, -0.4189176 ,  1.30756347,
             2.9947678 , -0.72225236, -1.34260555, -1.34023633, -1.35380493]))



## Solving $$\min ||y - (Ax)^2||^2, x \geq 0$$


    # generate the data, a linear model
    n = 10
    p = 20
    sigma = 0.00
    amat = np.absolute(npr.normal(0,1,(n,p)))
    bt0 = np.absolute(npr.normal(0,1,p)) * npr.binomial(1,3.0/p,p)
    mu0 = np.squeeze(np.dot(amat,bt0))
    y = mu0**2  + sigma*npr.normal(0,1,n)


    # solve using scipy.. fail!!
    def f(bt):
        yh = np.squeeze(np.dot(amat,bt))
        return y - yh**2
    
    res = spo.leastsq(f,np.absolute(npr.normal(0,1,p)))
    res[0],bt0


    # define out own function
    def pgauss_newton_it(y,amat,x0):
        yh = np.squeeze(np.dot(amat,x0))
        r = y-yh**2
        j = 2*np.dot(np.diag(yh),amat)
        y2 = r + np.squeeze(np.dot(j,x0))
        x1 = spo.nnls(j,y2)[0]
        return x1
    
    def pgauss_newton(y,amat,x0,nits):
        for ii in range(nits):
            x0 = pgauss_newton_it(y,amat,np.array(x0))
        return x0
    
    pgauss_newton(y,amat,np.absolute(npr.normal(0,1,p)),100),bt0




    (array([  1.16397109e-17,   0.00000000e+00,   0.00000000e+00,
              5.64872088e-16,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   5.25106298e-01,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   9.75842713e-17,
              0.00000000e+00,   4.48032099e-01,   0.00000000e+00,
              0.00000000e+00,   5.65562928e-03]),
     array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.5251063 ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.4480321 ,  0.        ,  0.        ,  0.00565563]))



## Solving $$\min ||y - ((Ax)^2 + c 1)||^2$$


    n = 100
    p = 10
    amat = npr.normal(0,1,(n,p))
    bt0 = npr.normal(0,1,p)
    c0 = npr.normal(0,1)
    mu0 = np.dot(amat,bt0)
    y = (mu0**2 + c0)


    # solve using scipy
    def f(c_bt):
        c = c_bt[0]
        bt = c_bt[1:]
        yh = np.squeeze(np.dot(amat,bt))
        return y - yh**2 - c
    
    res = spo.leastsq(f,npr.normal(0,1,p+1))
    res[0],c0,bt0




    (array([ 0.69734366, -0.19600593, -0.06909837,  0.94315796, -0.4189176 ,
             1.30756347,  2.9947678 , -0.72225236, -1.34260555, -1.34023633,
            -1.35380493]),
     0.6973436640384478,
     array([-0.19600593, -0.06909837,  0.94315796, -0.4189176 ,  1.30756347,
             2.9947678 , -0.72225236, -1.34260555, -1.34023633, -1.35380493]))




    # define our own function
    def cgauss_newton_it(y,amat,ex0):
        n = len(y)
        e0 = ex0[0]
        x0 = ex0[1:]
        yh = np.squeeze(np.dot(amat,x0))
        r = y-yh**2 - e0
        j_x = 2*np.dot(np.diag(yh),amat)
        j_c = np.ones((n,1))
        j = np.hstack([j_c,j_x])
        dex = nla.lstsq(j,r)[0]
        ex1 = ex0 + dex
        return ex1
    
    def cgauss_newton(y,amat,x0,nits):
        for ii in range(nits):
            x0 = cgauss_newton_it(y,amat,np.array(x0))
        return x0
    
    cgauss_newton(y,amat,npr.normal(0,1,p+1),100),c0,bt0




    (array([ 0.69734366,  0.19600593,  0.06909837, -0.94315796,  0.4189176 ,
            -1.30756347, -2.9947678 ,  0.72225236,  1.34260555,  1.34023633,
             1.35380493]),
     0.6973436640384478,
     array([-0.19600593, -0.06909837,  0.94315796, -0.4189176 ,  1.30756347,
             2.9947678 , -0.72225236, -1.34260555, -1.34023633, -1.35380493]))



## Solving $$\min ||y - ((Ax)^2 + c 1)||^2, x \geq 1$$


    # generate the data, a linear model
    n = 100
    p = 20
    sigma = 0.00
    amat = np.absolute(npr.normal(0,1,(n,p)))
    bt0 = np.absolute(npr.normal(0,1,p)) * npr.binomial(1,3.0/p,p)
    c0 = np.absolute(npr.normal(0,1))
    mu0 = np.squeeze(np.dot(amat,bt0))
    y = mu0**2  + c0 + sigma*npr.normal(0,1,n)


    # define out own function
    def cpgauss_newton_it(y,amat,ex0):
        yh = np.squeeze(np.dot(amat,ex0[1:]))
        r = y-yh**2-ex0[0]
        j = np.hstack([np.ones((n,1)),2*np.dot(np.diag(yh),amat)])
        y2 = r + np.squeeze(np.dot(j,ex0))
        ex1 = spo.nnls(j,y2)[0]
        return ex1
    
    def cpgauss_newton(y,amat,x0,nits):
        for ii in range(nits):
            x0 = cpgauss_newton_it(y,amat,np.array(x0))
        return x0
    
    zip(cpgauss_newton(y,amat,np.absolute(npr.normal(0,1,p+1)),100),np.hstack([c0,bt0]))




    [(0.29194747081154104, 0.2919474708115411),
     (4.1823028970688582e-17, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (1.1457505881812487e-17, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (7.8856798916369444e-17, 0.0),
     (1.0520949406543841e-19, 0.0),
     (0.0, 0.0),
     (0.27023950535857461, 0.27023950535857433),
     (0.53028433216638771, 0.53028433216638793),
     (2.6441452908426581e-18, 0.0)]



%% deterministic test
n = 100;
band = 5;
pp = n/band;
amat = zeros(n,pp);
for ii=1:pp;
    amat((ii-1)*band + (1:(2*band)),ii) = 1;
end
bt0 = zeros(pp,1);
bt0([1,5,7]) = 1;
c0 = 2;
mu0 = amat*bt0;
y = mu0.^2 + c0;

ex0 = ones(pp+1,1);
ex1,sses = cpgauss_newton(y,amat,ex0,100);
[ex1, [c0;bt0]]


    # resolve paradox with matlab
    n=40
    band = 4
    pp = n/band-1
    amat = np.zeros((n,pp))
    for ii in range(pp):
        amat[range(ii*band,(ii+2)*band),ii]=1;
    bt0 = np.zeros(pp)
    bt0[[1,3,4]] =1.0
    c0 = 2.0
    mu0 = np.squeeze(np.dot(amat,bt0))
    y = mu0**2  + c0
    ex0 = np.ones(pp+1)
    zip(cpgauss_newton(y,amat,ex0,100),np.hstack([c0,bt0]))




    [(2.0000000000000004, 2.0),
     (4.3000660053474428e-16, 0.0),
     (1.0000000000000007, 1.0),
     (4.3000660053474177e-16, 0.0),
     (1.0000000000000009, 1.0),
     (0.99999999999999911, 1.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (0.0, 0.0)]




    amat[:,1]




    array([ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])




    plt.scatter(range(n),y)
    plt.show()


    ex0 = np.ones(pp+1)
    ex1 = cpgauss_newton_it(y,amat,ex0)
    np.set_printoptions(suppress=True)
    print ex1

    [ 2.625   0.0625  1.0625  0.      1.125   0.6875  0.4375  0.375   0.5
      0.3125]



    ex0 = np.ones(pp+1)
    yh = np.squeeze(np.dot(amat,ex0[1:]))
    r = y-yh**2-ex0[0]
    j = np.hstack([np.ones((n,1)),2*np.dot(np.diag(yh),amat)])
    y2 = r + np.squeeze(np.dot(j,ex0))
    ex1 = spo.nnls(j,y2)[0]


    ex0




    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])




    amat[:,1]




    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])




    
