

    import numpy as np
    import numpy.linalg as npl
    import scipy as sp
    import scipy.stats as spst
    import scipy.special as sps
    import numpy.random as npr
    import matplotlib.pyplot as plt
    import cvxopt as cvx

    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)



    


    def bs4d(z): # standard basis function with derivatives
        i1 = np.array((z >= 0) & (z < 1),dtype = float)
        i2 = np.array((z >= 1) & (z < 2),dtype = float)
        i3 = np.array((z >= 2) & (z < 3),dtype = float)
        i4 = np.array((z >= 3) & (z < 4),dtype = float)
        c1 = z**3
        d1 = 3*z**2
        e1 = 6*z
        c2 = -3*z**3 + 12*z**2 - 12*z + 4
        d2 = -9*z**2 + 24*z - 12
        e2 = -18*z + 24
        c3 = 3*(z**3) - 24*(z**2) + 60*z - 44
        d3 = 9*(z**2) - 48*z + 60
        e3 = 18*z - 48
        c4 = (4-z)**3
        d4 = -3*(z-4)**2
        e4 = -6*(z-4)
        val0= i1*c1+i2*c2+i3*c3+i4*c4
        val1= i1*d1+i2*d2+i3*d3+i4*d4
        val2= i1*e1+i2*e2+i3*e3+i4*e4
        return val0,val1,val2
    
    def genspline(bt,scale,shift): # generates a spline from coefficients, extrapolating at endpoints
        nmax = len(bt)
        def f(x): # only evaluates at a single point
            z1 = scale*(x + shift) + 3
            z = z1
            if z1 < 3.5:
                z = 3.5
            if z1 > nmax:
                z = nmax
            h = np.floor(z)
            val0 = 0.0
            val1 = 0.0
            val2 = 0.0
            zr = z-h
            for j in range(-4,1):
                if (h+j) >=0 and (h+j) < len(bt):
                    cf = bt[h+j]
                    e0,e1,e2 = bs4d(zr-j)
                    val0 += cf*e0
                    val1 += cf*e1
                    val2 += cf*e2
            ex0 = val0 + val1*(z1-z) + .5*(z1-z)**2
            ex1 = val1 + val2*(z1-z)
            ex2 = val2
            return ex0,ex1*scale,ex2*(scale**2)
        return f
    
    def autospline(x,y): # generates a spline from uniform data
        mmax = max(x)
        mmin = min(x)
        nn = len(x)
        scale = (nn-1)/(mmax-mmin)
        shift = -mmin
        bt = np.dot(splinecoef(nn),y)
        return genspline(bt,scale,shift)
    
    def convspline(x,y): # generates a convex spline
        m=len(x)
        mmax = max(x)
        mmin = min(x)
        scale = (m-1)/(mmax-mmin)
        shift = -mmin
        atamat = cvx.matrix(np.dot(splinemat(m).T,splinemat(m))+ 1e-3 * np.diag(np.ones(m+2)))
        ytamat = cvx.matrix(-np.dot(y.T,splinemat(m)))
        hmat = cvx.matrix(-splinemat2(m))
        zerovec = cvx.matrix(np.zeros((m,1)))
        sol = cvx.solvers.qp(atamat,ytamat,hmat,zerovec)
        bt = np.squeeze(sol['x'])
        return genspline(bt,scale,shift)
    
    def splinemat(m):
        n = m+1
        mat= np.diag(1.* np.ones(n+2)) + np.diag(4.*np.ones(n+1),1) + np.diag(1.* np.ones(n),2)
        return mat[:(n-1),:(n+1)]
    
    def splinemat2(m):
        n = m+1
        mat= np.diag(6.* np.ones(n+2)) + np.diag(-12.*np.ones(n+1),1) + np.diag(6.* np.ones(n),2)
        return mat[:(n-1),:(n+1)]
    
    def splinecoef(m):
        return npl.pinv(splinemat(m))
    
    def quadinterp(x,y,dy,d2y): # an interpolating curve from points, first derivs and second derivs
        # x must be sorted
        def ff(z):
            if z < min(x):
                return y[0] + (z-x[0])*dy[0]+0.5*(z-x[0])**2*d2y[0],dy[0]+(z-x[0])*d2y[0],d2y[0]
            if z > max(x):
                return y[-1] + (z-x[-1])*dy[-1]+0.5*(z-x[-1])**2*d2y[-1],dy[-1]+(z-x[-1])*d2y[-1],d2y[-1]            
            ii = np.argwhere(x==z)
            if len(ii) > 0:
                ind = ii[0][0]
                return y[ind],dy[ind],d2y[ind]
            ii1 = np.argwhere(x < z)[-1][0]
            ii2 = np.argwhere(x > z)[0][0]
            x1 = x[ii1]
            y1 = y[ii1]
            dy1 = dy[ii1]
            d2y1=d2y[ii1]
            x2 = x[ii2]
            y2 = y[ii2]
            dy2 = dy[ii2]
            d2y2 = d2y[ii2]
            t = (z-x1)/(x2-x1)
            y3 = t*y2 + (1-t)*y1
            dy3 = t*dy2 + (1-t)*dy1 + (y2-y1)/(x2-x1)
            d2y3 = t*d2y2 + (1-t)*d2y1 + 2*(dy2-dy1)/(x2-x1)
            return y3,dy3,d2y3
        return ff


    def polyfunc(x,a,b,c):
        y = a*(x**3) + b*(x**2) + c*x
        dy = 3*a*(x**2) + 2*b*x + c
        d2y = 6*a*x + 2*b
        return y,dy,d2y


    az = npr.normal(0,1,20)
    bz = npr.normal(0,1,20)
    cz = npr.normal(0,1,20)
    ls = [0]*20
    xgrid = np.arange(0,1,0.001)
    for ii in range(20):
        y,dy,d2y = polyfunc(xgrid,az[ii],bz[ii],cz[ii])
        ls[ii] = quadinterp(xgrid,y,dy,d2y)
        #ls[ii]=convspline(xgrid,y)



    x = np.arange(0,1,0.1)
    a = 4.0
    b = 2.0
    c = 1.0
    y = a*(x**3) + b*(x**2) + c*x
    dy = 3*a*(x**2) + 2*b*x + c
    d2y = 6*a*x + 2*b
    ff = quadinterp(x,y,dy,d2y)
    z = np.arange(-1,2,0.01)
    plt.scatter(z,a*(z**3)+b*(z**2)+c*z)
    plt.scatter(z,np.array([ff(zz)[0] for zz in z]),color='red')
    plt.show()


    x = np.arange(-0.5,1,0.1)
    x2 = np.arange(-2,2,0.1)
    m=len(x)
    #y = -np.log(spst.ncx2.pdf(x=200.0,nc=x,df=3)) + 0.01*x**2
    y = x**3
    y20 = x2**3
    f = convspline(x,y)
    y2 = np.array([f(u)[2] for u in x2])
    #plt.scatter(x2,y20)
    plt.scatter(x2,y2,color='red')
    plt.show()

         pcost       dcost       gap    pres   dres
     0: -4.9893e-01 -6.4302e-01  2e+01  4e+00  2e+00
     1: -2.7225e-01 -9.5127e-01  1e+00  1e-01  8e-02
     2: -4.4868e-01 -6.0740e-01  2e-01  4e-16  3e-15
     3: -4.8716e-01 -5.0753e-01  2e-02  2e-16  8e-15
     4: -4.9550e-01 -5.0078e-01  5e-03  2e-16  1e-15
     5: -4.9705e-01 -4.9748e-01  4e-04  3e-16  7e-16
     6: -4.9732e-01 -4.9735e-01  3e-05  2e-16  3e-16
     7: -4.9733e-01 -4.9733e-01  4e-06  2e-16  8e-16
     8: -4.9733e-01 -4.9733e-01  1e-07  1e-16  1e-15
    Optimal solution found.



    m=len(x)
    mmax = max(x)
    mmin = min(x)
    scale = (m-1)/(mmax-mmin)
    shift = -mmin
    atamat = cvx.matrix(np.dot(splinemat(m).T,splinemat(m))+ 1e-3 * np.diag(np.ones(m+2)))
    ytamat = cvx.matrix(-np.dot(y.T,splinemat(m)))
    hmat = cvx.matrix(-splinemat2(m))
    zerovec = cvx.matrix(np.zeros((m,1)))
    sol = cvx.solvers.qp(atamat,ytamat,hmat,zerovec)
    bt = np.squeeze(sol['x'])

         pcost       dcost       gap    pres   dres
     0: -4.9893e-01 -6.4302e-01  2e+01  4e+00  2e+00
     1: -2.7225e-01 -9.5127e-01  1e+00  1e-01  8e-02
     2: -4.4868e-01 -6.0740e-01  2e-01  4e-16  3e-15
     3: -4.8716e-01 -5.0753e-01  2e-02  2e-16  8e-15
     4: -4.9550e-01 -5.0078e-01  5e-03  2e-16  1e-15
     5: -4.9705e-01 -4.9748e-01  4e-04  3e-16  7e-16
     6: -4.9732e-01 -4.9735e-01  3e-05  2e-16  3e-16
     7: -4.9733e-01 -4.9733e-01  4e-06  2e-16  8e-16
     8: -4.9733e-01 -4.9733e-01  1e-07  1e-16  1e-15
    Optimal solution found.



    nmax = len(bt)
    def f(x): # only evaluates at a single point
        z1 = scale*(x + shift) + 3
        z = z1
        if z1 < 0.5:
            z = 0.5
        if z1 > nmax-.5:
            z = nmax-.5
        h = np.floor(z)
        val0 = 0.0
        val1 = 0.0
        val2 = 0.0
        zr = z-h
        for j in range(-4,1):
            if (h+j) >=0 and (h+j) < len(bt):
                cf = bt[h+j]
                e0,e1,e2 = bs4d(zr-j)
                val0 += cf*e0
                val1 += cf*e1
                val2 += cf*e2
        ex0 = val0 + val1*(z1-z) + .5*(z1-z)**2
        ex1 = val1 + val2*(z1-z)
        ex2 = val2
        return ex0,ex1*scale,ex2*(scale**2)
    #return f


      File "<ipython-input-25-9bcbee23bf58>", line 25
        return f
    SyntaxError: 'return' outside function




    z1 = scale*(x + shift) + 3


    z1




    array([  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.,
            14.,  15.,  16.,  17.])




    
