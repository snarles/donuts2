

    import os
    import numpy as np
    import numpy.random as npr
    import numpy.linalg as nla
    import numpy.testing as npt
    import scipy as sp
    import scipy.stats as spst
    import scipy.special as sps
    import matplotlib.pyplot as plt
    import scipy.optimize as spo
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    os.chdir("..")
    import donuts.deconv.utils as du
    import donuts.deconv.ncx as ncx


    sps.sph_harm(1,1,.1,.1)




    array((-0.03431954573453935-0.003443440367396629j))




    bvecs = du.geosphere(4)
    n = np.shape(bvecs)[0]
    sgrid = du.geosphere(10)
    pp = np.shape(sgrid)[0]
    
    def plotz(zz=np.ones(n)):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(zz*bvecs[:,0],zz*bvecs[:,1],zz*bvecs[:,2])
        plt.show()
    def plotb(zz=np.ones(pp)):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(zz*sgrid[:,0],zz*sgrid[:,1],zz*sgrid[:,2])
        plt.show()
        
    def cart2sph(xyz): # rtp[:,1] is polar, rtp[:,2] is azimuthal
        rtp = np.reshape(xyz,(-1,3))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        r = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy),xyz[:,2])
        phi = np.arctan2(xyz[:,1],xyz[:,0])
        rtp = np.vstack([r,theta,phi]).T
        return rtp
    
    def sph2cart(rtp):
        rtp = np.reshape(rtp,(-1,3))
        st = np.sin(rtp[:,1])
        x = rtp[:,0]*st*np.cos(rtp[:,2])
        y = rtp[:,0]*st*np.sin(rtp[:,2])
        z = rtp[:,0]*np.cos(rtp[:,1])
        xyz = np.vstack([x,y,z]).T
        return xyz
    
    def test_sph2cart():
        xyz0 = du.geosphere(4)
        rtp = cart2sph(xyz0)
        xyz = sph2cart(rtp)
        npt.assert_almost_equal(xyz,xyz0,decimal=10)
        return
    
    geosphere = du.geosphere
    rand_ortho = du.rand_ortho
        
    
    def georandsphere(n,k):
        temp = [0]*k
        grid0 = geosphere(n)
        for ii in range(k):
            temp[ii] = np.dot(grid0,rand_ortho(3))
        ans = np.vstack(temp)
        return ans
    
    def real_sph_harm(m,n,rtp):
        rtp = np.reshape(rtp,(-1,3))
        p = rtp[:,2]
        t = rtp[:,1]
        if m==0:
            return np.sqrt(2)*sps.sph_harm(m,n,p,t).real
        if m > 0:
            return sps.sph_harm(m,n,p,t).real
        if m < 0:
            return np.sqrt(2)*sps.sph_harm(m,n,p,t).imag
        
    def randfunc(k,bandwidth):
        pos = du.normalize_rows(npr.normal(0,1,(k,3)))
        ws = np.ones((k,1))/k
        # generates a function on the sphere which is a mixture of "gaussians"
        def f(grid):
            y = np.squeeze(np.dot(np.exp(-(1-np.dot(grid,pos.T))**2/bandwidth),ws))
            return y
        return f
    
    def symrandfunc(k,bandwidth):
        pos = du.normalize_rows(npr.normal(0,1,(k,3)))
        ws = np.ones((k,1))/k
        # generates a function on the sphere which is a mixture of "gaussians"
        def f(grid):
            y = np.squeeze(np.dot(np.exp(-(1-np.dot(grid,pos.T))**2/bandwidth),ws)) + \
                np.squeeze(np.dot(np.exp(-(1-np.dot(-grid,pos.T))**2/bandwidth),ws))
            return y
        return f
    
    def rsh_basis(grid,n0):
        rtp = cart2sph(grid)
        temp = [0]*(n0+1)**2
        count = 0
        for n in range(n0+1):
            for m in range(-n,(n+1)):
                temp[count] = real_sph_harm(m,n,rtp)
                temp[count] = temp[count]/nla.norm(temp[count])
                count = count+1
        ans = np.vstack(temp).T
        return ans
    
    def rsh_basis_pos(grid,n0):
        rtp = cart2sph(grid)
        temp = [0]*(n0+1)**2
        count = 0
        for n in range(n0+1):
            for m in range(0,(n+1)):
                temp[count] = real_sph_harm(m,n,rtp)
                temp[count] = temp[count]/nla.norm(temp[count])
                count = count+1
        temp = temp[0:count]
        ans = np.vstack(temp).T
        return ans
    
    def test_rsh_basis():
        sgrid = du.georandsphere(5,8)
        xs = du.rsh_basis(sgrid,4)
        fudge = .5
        npt.assert_almost_equal(fudge*np.dot(xs.T,xs),fudge*np.eye(np.shape(xs)[1]),decimal=2)
        return


    reload(du)
    test_rsh_basis()


    sgrid = georandsphere(5,6)
    rtp = cart2sph(sgrid)
    f = symrandfunc(30.0,0.1)
    plotb(f(sgrid))


    np.set_printoptions(suppress=True)
    sgrid = georandsphere(6,1)
    rtp = cart2sph(sgrid)
    xs = rsh_basis(sgrid,8)
    xs_o = rsh_basis_pos(sgrid,8)
    q,r = nla.qr(xs)
    q_o, r = nla.qr(xs_o)
    #cc=np.dot(xs.T,xs)


    % matplotlib inline


    y = f(sgrid)
    coefs = np.squeeze(np.dot(q.T,y))
    yres = np.squeeze(np.dot(q,coefs))
    plt.scatter(y,yres); plt.show()


![png](theorySphHrm_files/theorySphHrm_7_0.png)



    y = f(sgrid)
    coefs = np.squeeze(np.dot(q_o.T,y))
    yres = np.squeeze(np.dot(q_o,coefs))
    plt.scatter(y,yres); plt.show()


![png](theorySphHrm_files/theorySphHrm_8_0.png)



    plt.scatter(range(np.shape(q)[1]),np.squeeze(np.dot(q.T,f(sgrid)))); plt.show()


![png](theorySphHrm_files/theorySphHrm_9_0.png)



    plt.scatter(range(np.shape(q_o)[1]),np.squeeze(np.dot(q_o.T,f(sgrid)))); plt.show()


![png](theorySphHrm_files/theorySphHrm_10_0.png)



    h11 = real_sph_harm(1,1,rtp)
    h12 = real_sph_harm(-1,2,rtp)
    np.dot(h11.T,h12),np.dot(h11.T,h11),np.dot(h12.T,h12)




    (-3.5648567431323386e-16, 144.03522349816566, 288.07044699632962)




    np.shape(h11)




    (3620,)




    




    1976.8318573239651




    
    
    sgrid = georandsphere(5,3)
    



    k =20.0
    bandwidth = .5
    f = randfunc(k,bandwidth)


    grid


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-123-0816eed6a2d8> in <module>()
    ----> 1 grid
    

    NameError: name 'grid' is not defined



    
