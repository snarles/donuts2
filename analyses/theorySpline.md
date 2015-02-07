

    import numpy as np
    import numpy.linalg as npl
    import scipy as sp
    import scipy.stats as spst
    import scipy.special as sps
    import numpy.random as npr
    import matplotlib.pyplot as plt
    from scipy import sparse


    def simplebs(i,k,x):
        if k==1:
            return np.array((x >= i) & (x < (i+1)),dtype = float)
        else:
            return (x-i)*simplebs(i,k-1,x) + (i+k-x)*simplebs(i+1,k-1,x)
        
    def bs3(i,x):
        i1 = np.array((x >= i) & (x < (i+1)),dtype = float)
        i2 = np.array((x >= (i+1)) & (x < (i+2)),dtype = float)
        i3 = np.array((x >= (i+2)) & (x < (i+3)),dtype = float)
        return i1 * (x-i)**2 + i2 * ((x-i)*(i+2-x)+(i+3-x)*(x-i-1)) + i3*(i+3-x)**2
    
    def bs4alt(i,x):
        i1 = np.array((x >= i) & (x < (i+1)),dtype = float)
        i2 = np.array((x >= (i+1)) & (x < (i+2)),dtype = float)
        i3 = np.array((x >= (i+2)) & (x < (i+3)),dtype = float)
        i4 = np.array((x >= (i+3)) & (x < (i+4)),dtype = float)
        c1 = (x-i)**3
        c2 = (x-i)*((x-i)*(i+2-x)+(i+3-x)*(x-i-1)) + (i+4-x)*(x-i-1)**2
        c3 = (x-i)*(i+3-x)**2 + (i+4-x)*((x-i-1)*(i+3-x) + (i+4-x)*(x-i-2))
        c4 = (i+4-x)**3
        return i1*c1+i2*c2+i3*c3+i4*c4
    
    def bs4(i,x):
        z = x-i
        i1 = np.array((z >= 0) & (z < 1),dtype = float)
        i2 = np.array((z >= 1) & (z < 2),dtype = float)
        i3 = np.array((z >= 2) & (z < 3),dtype = float)
        i4 = np.array((z >= 3) & (z < 4),dtype = float)
        c1 = z**3
        c2 = -3*z**3 + 12*z**2 - 12*z + 4
        c3 = 3*(z**3) - 24*(z**2) + 60*z - 44
        c4 = (4-z)**3
        return i1*c1+i2*c2+i3*c3+i4*c4
    
    def bs4e(z): # standard basis function
        i1 = np.array((z >= 0) & (z < 1),dtype = float)
        i2 = np.array((z >= 1) & (z < 2),dtype = float)
        i3 = np.array((z >= 2) & (z < 3),dtype = float)
        i4 = np.array((z >= 3) & (z < 4),dtype = float)
        c1 = z**3
        c2 = -3*z**3 + 12*z**2 - 12*z + 4
        c3 = 3*(z**3) - 24*(z**2) + 60*z - 44
        c4 = (4-z)**3
        return i1*c1+i2*c2+i3*c3+i4*c4
    
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
    
    def genspline(bt,scale,shift): # generates a spline from coefficients
        def f(x): # only evaluates at a single point
            z = scale*(x + shift) + 3
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
            return val0,val1,val2
        return f
    
    def autospline(x,y): # generates a spline from uniform data
        mmax = max(x)
        mmin = min(x)
        nn = len(x)
        scale = (nn-1)/(mmax-mmin)
        shift = -mmin
        bt = np.dot(splinecoef(nn),y)
        return genspline(bt,scale,shift)
    
    def numderiv(f,x,delta):
        return (f(x+delta)-f(x))/delta
    
    def numderiv2(f,x,delta):
        return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)


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


    x = np.arange(-3,10,0.1)
    f = genspline(np.array([1.0,0,0,0,0,2.0]),1.0,0.0)
    y = np.array([f(u)[0] for u in x])
    plt.scatter(x,y)
    plt.show()


    a = np.array([1.1,2.2,3.3,4.4])
    a[np.floor(3.5)]




    4.4000000000000004




    m=30
    x = np.arange(0,2,2.0/m)+1e-20
    y = np.log(spst.ncx2.pdf(x=20.0,nc=y,df=3)) + 0.01*x**2
    f = autospline(x,y)
    y2 = np.array([f(u)[0] for u in x])
    plt.scatter(x,y)
    plt.scatter(x,y2)
    plt.show()


    m=20
    x = np.arange(0,2,2.0/m)+1e-20
    y = x**2
    y = np.log(spst.ncx2.pdf(x=20.0,nc=y,df=3)) + 0.01*x**2
    #np.dot(np.dot(splinemat(m),splinecoef(m)),splinemat(m))
    bt = np.dot(splinecoef(m),y)
    sd = np.dot(splinemat2(m),bt)
    plt.scatter(x,sd)
    plt.show()
    np.where(sd < 0)




    (array([1, 3, 6, 8]),)




    simplebs(0,1,np.array([.5,1.5]))




    array([ 1.,  0.])




    x = np.array([.5,1.5])


    np.array((x >= 1) & (x <= 0),dtype=float)




    array([ 0.,  0.])




    x = np.arange(0,10,0.02)
    y = simplebs(4,4,x)
    y2 = bs4(4,x)
    plt.scatter(x,y-y2)
    plt.show()



    i = 2.0
    z = x-i
    c3 = (x-i)*(i+3-x)**2 + (i+4-x)*((x-i-1)*(i+3-x) + (i+4-x)*(x-i-2))
    c3a = 3*(z**3) - 24*(z**2) + 60*z - 44
    plt.scatter(x,c3-c3a)
    plt.show()


    z = np.arange(0,4,0.02)
    d2 = numderiv2(bs4e, z, 0.001)
    d2a = bs4d(z)[2]
    plt.scatter(z,d2a-d2)
    plt.show()



    bs4e(np.array([0,1,2,3,4],dtype=float))




    array([ 0.,  1.,  4.,  1.,  0.])




    
