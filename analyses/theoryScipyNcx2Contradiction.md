
# Try to approximate loss function using builtin ncx2

This code demonstrates a contradiction between two different ways of computing
the ncx2 log pdf using scipy built-in functions, [iv] the modified Bessel and
[hyp0f1] the limit hypergeometric function/also ncx2.logpdf


    import os
    import numpy as np
    import scipy as sp
    import scipy.stats as spst
    import scipy.special as sps
    import numpy.random as npr
    import matplotlib.pyplot as plt
    os.chdir("..")
    import donuts.deconv.splines as spl

    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)



    import donuts.deconv.ncx as ncx


    def logncx2pdf(x,df,nc):
        return -np.log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc) + logivy((df/2-1),np.sqrt(nc*x))
    
    def logivy(v,y):
        y = np.atleast_1d(y)
        ans = y
        ans[y < 500] = np.log(sps.iv(v,y[y < 500]))
        ans[y >= 500] = y[y >= 500] - np.log(2*np.pi*y[y >= 500])
        return ans


    df = 4
    x = 1000.0
    ncs = np.arange(0,10000.0,10.0)
    #plt.scatter(ncs,np.array([logncx2pdf(x,df,nc) for nc in ncs]))
    plt.scatter(ncs,logncx2pdf(x,df,ncs))
    plt.show()

    -c:2: RuntimeWarning: divide by zero encountered in divide



    nc = 600
    -np.log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc)
    logivy((df/2-1),np.sqrt(nc*x))
    v = (df/2-1)
    y = np.sqrt(nc*x)
    y - np.log(2*np.pi*y)




    766.10644970797489




    logncx2pdf(x,df,600)




    inf




    jj = np.arange(0.0,100.0,1.0)
    np.power(y[0]**2/4.0,jj)




    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])




    #def ncx2pdf(x,df,nc):
    #    return .5*np.exp(-(x+nc)/2)*np.power((x/nc),df/4 - .5)*sps.iv((df/2-1),np.sqrt(nc*x))
    
    def ncx2pdf(x,df,nc):
        return .5*np.exp(-(x+nc)/2.0)*np.power((x/nc),df/4 - .5)*sps.iv((df/2-1),np.sqrt(nc*x))
    
    def ncx2pdf2(x,df,nc):
        return np.exp(-(x+nc)/2)*np.power(x,df/2-1)*np.power(.5,df/2)/sps.gamma(df/2) * sps.hyp0f1(df/2,nc*x/4.0)
    
    
    
    def ivy_sub(v,y):
        jj = np.arange(0.0,100.0,1.0)
        return ((y/2)**v) * sum(np.power(y**2/4.0,jj)/(sps.gamma(jj+1)*sps.gamma(v+jj+1)))
    def ivy(v,y):
        return np.array([ivy_sub(v,yy) for yy in y])
    
    def logivy(v,y):
        return np.log(sps.iv(v,y))
    
    def logivy_a(v,y): # asymptotic approximation
        return y - np.log(2*np.pi*y)
    
    def logivy_a2(v,y): # asymptotic approximation
        return -sps.gammaln(v+1) + v*np.log(y/2)
    
    df = 10
    nc = 7000
    x = np.arange(0,100,1.0)
    
    v = df/2 - 1
    y = np.sqrt(nc*x)
    
    
    
    #zip(spst.ncx2.pdf(x,df,nc),ncx2pdf(x,df,nc))
    #zip(sps.iv(v,y),ivy(v,y))
    #plt.scatter(x,spst.ncx2.pdf(x,df,nc))
    #plt.scatter(x,ncx2pdf2(x,df,nc),color="red")
    #plt.show()
    
    #plt.scatter(y,logivy(v,y))
    #plt.scatter(y,logivy_a(v,y),color="red")
    #plt.scatter(y,logivy_a2(v,y),color="green")
    #plt.show()
    
    
    v = 1.0
    ys = np.arange(0.0,1000.0,10.0)
    plt.scatter(ys,np.array([logivy_a(v,y) for y in ys]),color="red")
    #plt.scatter(ys,np.array([logivy_a2(v,y) for y in ys]),color="green")
    plt.scatter(ys,np.array([logivy(v,y) for y in ys]))
    plt.show()


                
                

    

    -c:24: RuntimeWarning: divide by zero encountered in log



    import timeit
    %timeit ncx2pdf(x,df,nc)

    10000 loops, best of 3: 83.9 µs per loop



    %timeit ncx2pdf2(x,df,nc)

    1000 loops, best of 3: 628 µs per loop



    
    def sq(x):
        return x**2
    def logderivs(f,df,d2f,x):
        return [np.log(f(x)),df(x)/f(x), -(df(x)/f(x))**2 + d2f(x)/f(x)]
    def x2derivs(f,df,d2f,x):
        return [f(x**2),2*x*df(x**2),2*df(x**2) + ((2*x)**2)*d2f(x**2)]
    def logx2derivs(f,df,d2f,x):
        return [np.log(f(x**2)),
                2*x*df(x**2)/f(x**2),
                2*df(x**2)/f(x**2) + ((2*x)**2)*(-(df(x**2)/f(x**2))**2 + d2f(x**2)/f(x**2))]


    def logncx2pdf(x,df,nc):
        return -log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc) + logivy((df/2-1),np.sqrt(nc*x))
    
    def logivy(v,y):
        if v < 500:
            return np.log(sps.iv(v,y))
        else:
            return y - np.log(2*np.pi*y)
    
    def numderiv(f,x,delta):
        return (f(x+delta)-f(x))/delta
    def numderiv2(f,x,delta):
        return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)
    
    def logncx2(n,nc,x):
        def ff(x):
            return spst.ncx2.logpdf(x,n,nc)
        l0 =ff(x)
        l1=numderiv(ff,x,1e-3)
        l2=numderiv2(ff,x,1e-3)
        return l0,l1,l2
    
    def losschi2(n,mu,x):
        nc = mu**2.0
        l0,l1,l2 = logncx2(n,nc,x)
        return -1.0*np.array([l0,2*mu*l1,2*l1 + ((2*mu)**2)*l2])
    
    def genconvexchi2(n,x): # returns a function which is the convex relaxation of losschi2
        mmax = np.sqrt(x)*3
        mugrid = np.arange(-mmax,mmax,mmax/100)
        evalu = losschi2(n,mugrid,x)
        
        def convloss(mu):
            if mu > muinf:
                return losschi2(n,mu,x)
            else:
                return [res[0] + res[1]*(mu-muinf)+res[2]*(mu-muinf)**2,res[1] + res[2]*(mu-muinf),res[2]]
        return convloss


    n=10.0
    mu=100.0
    nc=mu**2
    x=1000.0
    
    mmax = np.sqrt(x)*3
    mugrid = np.arange(-mmax,mmax,mmax/100)
    evalu = losschi2(n,mugrid,x)
    evalu




    array([[ -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,   5.14575981e+001,
              6.10080847e+001,   7.14493137e+001,   8.27803840e+001,
              9.50002561e+001,   1.08107723e+002,   1.22101369e+002,
              1.36979525e+002,   1.52740198e+002,   1.69380983e+002,
              1.86898943e+002,   2.05290438e+002,   2.24550881e+002,
              2.44674379e+002,   2.65653184e+002,   2.87476820e+002,
              3.10130637e+002,   3.33593241e+002,   3.57831596e+002,
              3.82790672e+002,   4.08368185e+002,   4.34337927e+002,
              4.60003059e+002,   4.79012769e+002,   4.60003059e+002,
              4.34337927e+002,   4.08368185e+002,   3.82790672e+002,
              3.57831596e+002,   3.33593241e+002,   3.10130637e+002,
              2.87476820e+002,   2.65653184e+002,   2.44674379e+002,
              2.24550881e+002,   2.05290438e+002,   1.86898943e+002,
              1.69380983e+002,   1.52740198e+002,   1.36979525e+002,
              1.22101369e+002,   1.08107723e+002,   9.50002561e+001,
              8.27803840e+001,   7.14493137e+001,   6.10080847e+001,
              5.14575981e+001,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308,  -1.79769313e+308,
             -1.79769313e+308,  -1.79769313e+308],
           [  0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,   0.00000000e+000,
              0.00000000e+000,   0.00000000e+000,  -6.68749733e+000,
             -7.02285652e+000,  -7.30129475e+000,  -7.52281199e+000,
             -7.68740824e+000,  -7.79508351e+000,  -7.84583778e+000,
             -7.83967108e+000,  -7.77658338e+000,  -7.65657470e+000,
             -7.47964502e+000,  -7.24579436e+000,  -6.95502270e+000,
             -6.60733005e+000,  -6.20271639e+000,  -5.74118173e+000,
             -5.22272606e+000,  -4.64734935e+000,  -4.01505158e+000,
             -3.32583270e+000,  -2.57969258e+000,  -1.77663085e+000,
             -9.16645882e-001,  -2.39651855e-013,   9.16645882e-001,
              1.77663085e+000,   2.57969258e+000,   3.32583270e+000,
              4.01505158e+000,   4.64734935e+000,   5.22272606e+000,
              5.74118173e+000,   6.20271639e+000,   6.60733005e+000,
              6.95502270e+000,   7.24579436e+000,   7.47964502e+000,
              7.65657470e+000,   7.77658338e+000,   7.83967108e+000,
              7.84583779e+000,   7.79508350e+000,   7.68740825e+000,
              7.52281198e+000,   7.30129475e+000,   7.02285653e+000,
              6.68749732e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000,  -0.00000000e+000,
             -0.00000000e+000,  -0.00000000e+000],
           [              nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,   6.38391238e-001,
              6.26687058e-001,   6.18992063e-001,   6.15038582e-001,
              6.14125521e-001,   6.16273741e-001,   6.20968774e-001,
              6.28748797e-001,   6.39121432e-001,   6.51845590e-001,
              6.66931876e-001,   6.84097844e-001,   7.03122419e-001,
              7.24140569e-001,   7.46692956e-001,   7.70690156e-001,
              7.96008252e-001,   8.22526389e-001,   8.49987403e-001,
              8.78267413e-001,   9.07199839e-001,   9.36608966e-001,
              9.66263568e-001,   9.92000004e-001,   9.66263568e-001,
              9.36608966e-001,   9.07199839e-001,   8.78267413e-001,
              8.49987403e-001,   8.22511655e-001,   7.96028307e-001,
              7.70690156e-001,   7.46692956e-001,   7.24161033e-001,
              7.03221463e-001,   6.84038908e-001,   6.66828126e-001,
              6.51765373e-001,   6.39029346e-001,   6.28801184e-001,
              6.20909635e-001,   6.15478115e-001,   6.14125521e-001,
              6.14547454e-001,   6.19172552e-001,   6.27083234e-001,
              6.38607743e-001,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan,               nan,
                          nan,               nan]])



Tests


    n=10.0
    mu=100.0
    nc=mu**2
    x=10000.0
    def f(nc):
        return logncx2(n,nc,x)[0]
    
    def f0(nc):
        return np.log(spst.ncx2.pdf(x,n,nc))
    
    
    print([f0(nc),numderiv(f0,nc,0.001),numderiv2(f0,nc,0.001)])
    print(logncx2(n,nc,x))
    #print(logncx2prox(n,nc,x))
    print([f(nc),numderiv(f,nc,0.01),numderiv2(f,nc,0.01)])

    [inf, nan, nan]
    (-175.74924925918066, -0.088709228134640189, -4.1064781022108849e-05)
    [-175.74924925918066, -0.088709433660483228, -4.1188741306541488e-05]



    def g(mu):
        return logncx2(n,mu**2,x)[0]
    
    print(-1.0*losschi2(n,mu,x))
    print([g(mu),numderiv(g,mu,0.01),numderiv2(g,mu,0.01)])

    [ -4.59269912e+03  -8.00208798e+01  -1.19977283e+00]
    [-4592.6991236097974, -80.026878622993536, -1.1997727960988414]



    n=10.0
    x=3000.0
    
    clos = genconvexchi2(n,x)
    mu = 0.0
    print(clos(mu))
    print(losschi2(n,mu,x))

    [1484.4136963225424, -50.581928480787425, 0.0093565037358018799]
    [ 1474.61831946    -0.          -299.        ]



    n=10.0
    x=8000.0
    
    mugrid = np.arange(-200.0,200.0,1.0)
    pts = [losschi2(n,mu,x)[0] for mu in mugrid]
    clos = genconvexchi2(n,x)
    #pts2 = [clos(mu)[0] for mu in mugrid]
    #pts = [np.log(spst.ncx2.pdf(x=15.0,df=10.0,nc=mu**2)) for mu in mugrid]
    plt.scatter(mugrid,pts)
    #plt.scatter(mugrid,pts2)
    plt.show()

    -c:4: RuntimeWarning: overflow encountered in double_scalars
    -c:4: RuntimeWarning: invalid value encountered in double_scalars



    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-3-26faeddc8049> in <module>()
          4 mugrid = np.arange(-200.0,200.0,1.0)
          5 pts = [losschi2(n,mu,x)[0] for mu in mugrid]
    ----> 6 clos = genconvexchi2(n,x)
          7 #pts2 = [clos(mu)[0] for mu in mugrid]
          8 #pts = [np.log(spst.ncx2.pdf(x=15.0,df=10.0,nc=mu**2)) for mu in mugrid]


    NameError: name 'genconvexchi2' is not defined



    zip(mugrid,pts)
    mu=33
    losschi2(n,mu,x)[0]




    101.2432780953945




    mu = 33.0
    nc = mu**2
    eps = 30.0
    if nc < 1e-20:
        nc = 1e-20
    a = n/2.0
    temp0 = -(nc+x)/2.0 - a*np.log(2.0) + (a-1)*np.log(x)
    
    # adaptive choose sum indices
    def f0(z):
        return -sps.gammaln(a + z) + z*np.log(nc*x/4) - sps.gammaln(z+1)
    lb,ub = boundseq(f0,eps)
    iz0 = np.arange(lb,ub,1.0)
    #iz0 = np.arange(0,1000,1.0)
    tempA = -sps.gammaln(a + iz0) + iz0*np.log(nc*x/4)
    s0 = logsumexp(temp0+tempA - sps.gammaln(iz0+1))
        


    s0




    -101.2432780953945




    lb,ub = boundseq(f0,eps)
    (lb,ub)




    (448.0, 603.0)




    maxseq(f0)




    512.0




    f=f0
    diff = 1
    ub=1
    eps=0.1
    while diff > 0:
        ub=2*ub
        diff = f(ub*(1+eps))-f(ub)


    ub




    1024




    
    # bisection search
    lb=0
    x = (ub+lb)/2.0
    while (f(ub)-f(lb)) > eps:
        x = (ub+lb)/2.0
        diff = f(x*(1+eps))-f(x)
        if diff < 0:
            ub = x
        if diff >= 0:
            lb = x
    x




    512.0




    (lb,ub)




    (0, 1024)




    (f(ub)-f(lb))




    -51.654878105960393




    f0(513.0)-f0(512.0)




    0.71930450167565141




    plt.scatter(iz0,f0(iz0))
    plt.show()


    plt.scatter(iz0,tempA- sps.gammaln(iz0+1))
    plt.show()


    eps = 30.0
    if nc < 1e-20:
        nc = 1e-20
    a = n/2.0
    temp0 = -(nc+x)/2.0 - a*np.log(2.0) + (a-1)*np.log(x)
    
    # adaptive choose sum indices
    def f0(z):
        return -sps.gammaln(a + z) + z*np.log(nc*x/4) - sps.gammaln(z+1)
    lb,ub = boundseq(f0,eps)


    (lb,ub)




    (0.0, 11.0)




    f = f0
    eps=20.0
    x0=maxseq(f)
    lb = lbseq(f,x0,eps)
    ub = ubseq(f,x0,eps)
    [x0,lb,ub]




    [1.0, 0.0, 1.0]




    
    n=10.0
    nc=0.0
    x=2.0
    if True:
        a = n/2.0
        iz = np.arange(0,100,1)
        temp0 = -(nc+x)/2.0 - a*np.log(2.0) + (a-1)*np.log(x)
        temp = -sps.gammaln(a + iz) + iz*np.log(nc*x/4)
        s0 = sum(np.exp(temp - sps.gammaln(iz+1)))
        s1 = sum(np.exp(temp - sps.gammaln(iz)))
        tempalt = -sps.gammaln(a + iz) + iz*np.log(x/4) - sps.gammaln(iz)
        #return np.exp(temp0)*(-.5*s0 + (1.0/nc)*s1)
    np.exp(temp0)*(-.5*s0 + sum(np.exp(tempalt)))




    nan




    -.5*s0 




    nan




    losschi2(10.0,3.0,2.0)




    [-8.5291642798643839, -2.7191735130423904, -2.2280755037050906]




    losschi2(10.0,1e-20,2.0)




    [-4.8712010109078907, -7.8272552047574014e-21, -0.78272552047574018]




    


    mugrid = np.arange(0,5,0.01)
    pts = [losschi2(10.0,mu,2.0)[2] for mu in mugrid]
    #pts = [np.log(spst.ncx2.pdf(x=15.0,df=10.0,nc=mu)) for mu in mugrid]
    plt.scatter(mugrid,pts)
    plt.show()


    
