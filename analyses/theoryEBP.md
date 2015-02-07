
Elastic basis pursuit


    import os
    import numpy as np
    import scipy as sp
    import scipy.stats as spst
    import scipy.special as sps
    import numpy.random as npr
    import matplotlib.pyplot as plt
    import numpy.random as npr
    import scipy.optimize as spo
    
    os.chdir("..")
    import donuts.deconv.splines as spl

    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)



    #ncx code
    
    def numderiv2(f,x,delta):
        return (f(x+delta)+f(x-delta)-2*f(x))/(delta**2)
    
    def numderiv(f,x,delta):
        return (f(x+delta)-f(x))/delta
    
    def logivy(v,y):
        y = np.atleast_1d(y)
        ans = np.array(y)
        ans[y < 500] = np.log(sps.iv(v,y[y < 500]))
        ans[y >= 500] = y[y >= 500] - np.log(2*np.pi*y[y >= 500])
        return ans
    
    def logncx2pdf_x(x,df,nc): #only x varies
        if nc==0:
            return spst.chi2.logpdf(x,df)
        else:
            return -np.log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc) + logivy((df/2-1),np.sqrt(nc*x))
    
    def logncx2pdf_nc(x,df,nc0): #only nc varies
        nc0 =np.atleast_1d(nc0) 
        nc = np.array(nc0)
        nc[nc0 < 1e-5] = 1e-5
        ans= -np.log(2.0) -(x+nc)/2.0 + (df/4 - .5)*np.log(x/nc) + logivy((df/2-1),np.sqrt(nc*x))
        return ans
    
    def convex_nc_loss(x,df):
        def ff(mu):
            return -logncx2pdf_nc(x,df,mu**2)
        def f2(mu):
            return numderiv2(ff,mu,1e-3) - 1e-2
        mugrid = np.arange(0.0,2*df,df*0.01)
        res = np.where(f2(mugrid) < 1e-2)[0]
        if len(res) > 0:
            imin = np.where(f2(mugrid) < 1e-2)[0][-1]
            muinf = mugrid[imin]
        else:
            muinf = 0.0
        val = ff(muinf)
        dval = numderiv(ff,muinf,1e-3)
        d2val = 1e-2
        #print(muinf)
        def cff(mu):
            mu = np.atleast_1d(mu)
            ans = np.array(mu)
            ans[mu > muinf] = -logncx2pdf_nc(x,df,mu[mu > muinf]**2)
            ans[mu <= muinf] = val + (mu[mu <= muinf]-muinf)*dval + .5*d2val*(mu[mu <= muinf]-muinf)**2
            return ans
        return cff


    def pruneroutine(v1,v2):
        # finds the minimum convex combination of v1 and v2 so that exactly one element is nonpositive (zero)
        # v1 is nonnegative
        v1 = np.atleast_1d(v1)
        v2 = np.atleast_1d(v2)
        assert min(v1)>=0
        if min(v2) >=0:
            return v2,-1
        else:
            mina = np.array(v2)*0
            mina[v1 != v2]= -v1[v1 != v2]/(v2[v1 != v2]-v1[v1 != v2])
            ans = (1-mina)*v1 + mina*v2
            assert min(ans) >= -1e-15
            mina[v2 >= 0] = 1e99
            mina[v2==v1] = 1e99
            a = min(mina)
            assert a <= 1
            assert a >= 0
            o = np.where(mina == a)[0][0]
            ans = (1-a)*v1 + a*v2
            assert min(ans) >= -1e-15
            ans[ans <0] =0
            return ans,o
    
    def subrefitting(amat,ls,x0,newind): # refit x0 so that grad(x0)=0 where x0 positive
        oldx0 = np.array(x0)
        s = np.zeros(p,dtype=bool)
        s[np.squeeze(x0) > 1e-20] = True
        s[newind] = True
        amat2 = amat[:,s]
        x02 = np.array(x0[s])
        x02 = bfgssolve(amat2,ls,np.array(x02),-1.0)[0]
        oldx02 = np.array(x02)
        x0[~s] = 0.0
        x0[s]=x02
        flag = min(x0) < 0
        x0 = pruneroutine(oldx0,np.array(x0))[0]
        while flag:
            oldx0 = np.array(x0)
            s = np.zeros(p,dtype=bool)
            s[np.squeeze(x0) > 1e-20] = True
            amat2 = amat[:,s]
            x02 = np.array(x0[s])
            x02 = bfgssolve(amat2,ls,np.array(x02),-1.0)[0]
            x0[~s] = 0.0
            x0[s]=x02
            flag = min(x0) < 0
            x0new = np.array(x0)
            #print(min(x0))
            x0 = pruneroutine(oldx0,np.array(x0))[0]
        return x0
    
    def ebp(amat,ls,x0): # ls is a list of loss functions, x0 is initial guess
        x0seq = [np.array(x0)]
        newind = np.where(x0==max(x0))[0]
        p = np.shape(amat)[1]
        flag = True
        count = 0
        while flag:
            count = count + 1
            # **** refitting step ****
            x0 = subrefitting(amat,ls,np.array(x0),newind)
            # next candidate step
            yh = np.dot(amat,x0)
            rawg = np.array([ls[i](yh[i])[1] for i in range(n)])
            g = np.dot(rawg.T,amat)
            if min(g) > -1e-5:
                flag=False
            else:
                newind = np.where(g==min(g))[0][0]
            if count > 1000:
                flag = False
            x0seq = x0seq + [np.array(x0)]
        return x0,x0seq
    
    def bfgssolve(amat,ls,x0,lb=0.0): # use LBFS-G to solve
        def f(x0):
            yh = np.dot(amat,x0)
            return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
        def fprime(x0):
            yh = np.dot(amat,x0)
            rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
            return np.dot(rawg.T,amat)
        bounds = [(lb,100.0)] * len(x0)
        res = spo.fmin_l_bfgs_b(f,np.squeeze(x0),fprime=fprime,bounds=bounds)
        return res
        
    
    def ncxlosses(df,y):
        n = len(y)
        ans = [0.] * n
        for ii in range(n):
            x = y[ii]
            mmax = np.sqrt(x)*3
            mugrid = np.arange(0,mmax,mmax/100)
            clos = convex_nc_loss(n,x)
            pts = clos(mugrid)
            f = spl.convspline(mugrid,pts)
            ans[ii]=f
        return ans


    n = 20
    p = 500
    amat = np.absolute(npr.normal(0,1,(n,p)))
    bt0 = np.zeros((p,1))
    bt0[:2] = 1
    df = 10
    mu = np.dot(amat,bt0)
    ysq = spst.ncx2.rvs(df,mu**2)
    ls = ncxlosses(df,ysq)
    
    def f(x0):
        yh = np.dot(amat,x0)
        return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
    def fprime(x0):
        yh = np.dot(amat,x0)
        rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
        return np.dot(rawg.T,amat)


    bt = spo.nnls(amat,np.squeeze(np.sqrt(ysq)))[0]
    #print(f(bt))
    res = bfgssolve(amat,ls,np.array(bt),0.0)
    x0 = res[0]
    (f(x0),sum(x0 > 0))




    (59.093333593462773, 32)




    resebp = ebp(amat,ls,np.array(bt))
    x0 = resebp[0]
    (f(x0),sum(x0 > 0))




    (59.090007466075789, 19)




    

Testing EBP


    x0 = np.array(bt)
    x0seq = [np.array(x0)]
    newind = np.where(x0==max(x0))[0]
    newinds = [newind]
    p = np.shape(amat)[1]
    flag = True
    count = 0
    f(x0)




    74.579729607938319




    count = count + 1
    # **** refitting step ****
    x0 = subrefitting(amat,ls,np.array(x0),newind)
    # next candidate step
    yh = np.dot(amat,x0)
    rawg = np.array([ls[i](yh[i])[1] for i in range(n)])
    g = np.dot(rawg.T,amat)
    if min(g) > -1e-5:
        flag=False
    else:
        newind = np.where(g==min(g))[0][0]
        newinds = newinds + [newind]
    if count > 1000:
        flag = False
    x0seq = x0seq + [np.array(x0)]
    (f(x0),sum(x0 > 0),min(g),count,flag)




    (59.090007454007385, 19, 9.7016050047575832e-05, 46, False)




    (f(x0seq[33]),newinds[33])




    (59.767210298716819, 42)




    indark = 33
    x0ark = x0seq[indark]
    newindark = newinds[indark]


    (f(x0ark),min(x0ark))




    (59.767210298716819, -7.161243359447845e-34)




    x0 = np.array(x0ark)
    newind = newindark

Testing subroutine


    #x0 = np.array(x0ark)
    #newind = newindark
    
    x0 = np.array(bt)
    print(f(x0))
    newind = np.where(x0==max(x0))[0]
    oldx0 = np.array(x0)
    s = np.zeros(p,dtype=bool)
    s[np.squeeze(x0) > 1e-20] = True
    s[newind] = True
    amat2 = amat[:,s]
    x02 = np.array(x0[s])
    x02 = bfgssolve(amat2,ls,np.array(x02),-1.0)[0]
    oldx02 = np.array(x02)
    x0[~s] = 0.0
    x0[s]=x02
    flag = min(x0) < 0
    x0 = pruneroutine(oldx0,np.array(x0))[0]
    (f(x0),flag)

    74.5797296079





    (74.026986542777422, True)




    oldx0 = np.array(x0)
    s = np.zeros(p,dtype=bool)
    s[np.squeeze(x0) > 1e-20] = True
    amat2 = amat[:,s]
    x02 = np.array(x0[s])
    x02 = bfgssolve(amat2,ls,np.array(x02),-1.0)[0]
    x0[~s] = 0.0
    x0[s]=x02
    flag = min(x0) < 0
    x0new = np.array(x0)
    print(min(x0))
    x0 = pruneroutine(oldx0,np.array(x0))[0]
    (f(x0),flag)

    -1.0





    (66.159118489625186, True)




    x0new[s]




    array([-0.10506345,  0.5501807 ,  0.11715688, -0.08283377,  0.17583834,
           -0.29381685, -0.50938756,  0.84747827, -0.30582553,  0.29435681,
           -0.29329957,  1.85197973,  0.19563369, -0.81881465,  0.04033095,
            0.39345424, -1.        ,  0.23736429,  0.29460637,  1.07917614])




    x0[s]




    array([  3.09443836e-001,   3.39179106e-002,   2.22691967e-003,
             1.14636390e-001,   8.97754569e-002,   2.72756942e-001,
             1.56464762e-001,   9.05035198e-002,   2.73976169e-194,
             6.66546422e-003,   1.25249114e-001,   1.36328466e-001,
             2.32809080e-002,   4.82349166e-001,   8.38789646e-002,
             3.35434493e-002,   6.21468144e-001,   1.39023728e-001,
             5.44056749e-001,   4.71546615e-001])




    resebp = ebp(amat,ls,np.array(bt))
    x0 = resebp[0]
    f(x0)


    [sum(resebp[0] > 0),sum(res[0] > 0)]




    [17, 18]




    fprime(x02)[x02 > 0]




    array([  2.31243524e-05,   1.79270371e-05,   1.53864876e-06,
             1.12106956e-05])




    fprime(res[0])




    array([  3.99364630e-01,  -3.09226073e-04,   7.73182357e-01,
             1.74471296e+00,   1.03643085e+00,   2.10586148e-05,
             1.13097663e+00,   3.56040954e+00,   1.53363885e+00,
             6.88656365e-01,   5.70117040e-02,   2.44839121e+00,
             1.93246482e+00,   1.28812489e+00,   3.75937363e-01,
             2.50614167e+00,   7.14514672e-01,   7.18007471e-05,
             7.80825760e-01,   1.80548021e+00,   1.06478627e+00,
             9.90569691e-01,   1.33522849e+00,   1.02688863e+00,
             1.41995241e+00,   3.50632248e-01,   1.88792772e-01,
             1.21341736e+00,   9.43430087e-01,   9.44738604e-01,
             1.35952557e+00,   9.23825914e-01,   1.63315541e+00,
             1.41722389e+00,   7.24538326e-01,   1.42027495e+00,
             1.37027386e+00,   2.34512129e+00,   1.26214092e+00,
             4.32565809e-01,   1.61736499e+00,   1.96060367e+00,
             1.86992715e+00,   1.39756294e+00,  -1.00155566e-04,
            -1.28536442e-04,   1.10912127e-01,   7.07345295e-01,
             7.80844747e-01,   1.72308676e-01])




    resebp[1]




    [array([ 0.        ,  1.0664185 ,  2.16817493,  0.        ,  0.86690074,
             0.29303668,  0.        ,  0.        ,  0.15695727,  0.        ]),
     array([ 0.        ,  1.0664185 ,  2.16817493,  0.        ,  0.86690074,
             0.29303668,  0.        ,  0.        ,  0.15695727,  0.        ])]




    x0 = np.array(bt)
    p = np.shape(amat)[1]
    flag = True
    s = np.zeros(p,dtype=bool)
    s[np.squeeze(x0) > 1e-20] = True
    x0seq = [np.array(x0)]


    amat2 = amat[:,s]
    x02 = np.array(x0[s])
    res02 = bfgssolve(amat2,ls,x02,0.0)
    x02 =res02[0]
    x0[s]=x02
    
    yh = np.dot(amat,x0)
    rawg = np.array([ls[i](yh[i])[1] for i in range(n)])
    
    g = np.dot(rawg.T,amat)
    print([loss(x0),min(g),x0])
    
    if min(g) > -1e-5:
        flag=False
    else:
        ind = np.where(g==min(g))[0][0]
        s[ind]=True
        if x0[ind] < 1e-20:
            x0[ind]=1e-5
    x0seq = x0seq + [np.array(x0)]


    [13.761844147550029, -1.8228589925418447e-05, array([  1.34832349e-05,   5.44153734e-01,   1.29704312e+00,
             0.00000000e+00,   2.76881515e-07,   8.91047117e-01,
             0.00000000e+00,   0.00000000e+00,   3.29950258e-01,
             9.82124967e-01])]



    [13.415306944768647, -0.0051969849982899143]


    x0a = np.array(x0)


    x02




    array([ 1.02209522,  0.98257249,  2.16381201,  0.47245812])




    pruneroutine(x02old,x02)




    (array([ 1.01623394,  0.97027142,  2.15458253,  0.47730043]), 0)




    amat2




    array([[ 0.78704991,  0.72717056,  0.34022448],
           [ 0.57777142,  0.66648198,  0.58797243],
           [ 0.95991539,  0.03975101,  0.27871945],
           [ 1.2130095 ,  1.02209947,  0.21570625],
           [ 0.34107076,  0.596097  ,  1.29039437]])




    amat2 = amat[:,s]
    x02 = x0[s]
    x02old = x02
    x02,ls0 = gd(amat2,ls,x02)
    x02,ls1 = gd(amat2,ls,x02)
    while (ls0 - ls1) > 1e-3:
        ls0 = ls1
        x02,ls1 = gd(amat2,ls,x02)


    x0seq




    [array([ 2.89851685,  0.42807301,  0.        ,  1.24165962,  0.        ,
             0.16532922,  0.        ,  0.        ,  0.        ,  0.        ]),
     array([  2.78929675e+00,   3.60419061e-01,   0.00000000e+00,
              1.27441727e+00,   0.00000000e+00,   2.77555756e-17,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00]),
     array([ 2.78929675,  0.36041906,  0.        ,  1.27441727,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ])]




    bt




    array([ 2.89851685,  0.42807301,  0.        ,  1.24165962,  0.        ,
            0.16532922,  0.        ,  0.        ,  0.        ,  0.        ])




    x0




    array([  2.78929675e+00,   3.60419061e-01,   0.00000000e+00,
             1.27441727e+00,   0.00000000e+00,   2.77555756e-17,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00])




    mugrid=np.arange(0,3,.1)
    mkmk = [ls[0](x)[0] for x in mugrid]
    plt.scatter(mugrid,mkmk)
    plt.show()


    x0 = bt
    def f(x0):
        yh = np.dot(amat,x0)
        return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
    def fprime(x0):
        yh = np.dot(amat,x0)
        rawg= np.array([ls[i](yh[i])[1] for i in range(len(yh))])
        return np.dot(rawg.T,amat)
    bounds = [(0.0,100.0)] * len(x0)
    res = spo.fmin_l_bfgs_b(f,np.squeeze(x0),fprime=fprime,bounds=bounds)



    res




    (array([ 2.0186227 ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.61353646,  3.08337257,  0.16916887,  0.        ,  0.        ]),
     14.400339341247355,
     {'funcalls': 32,
      'grad': array([  2.03732614e-05,   5.04307039e-02,   7.85928734e-03,
               3.29657294e-02,   3.78908633e-02,   2.39820571e-05,
              -9.32518397e-06,   1.63477234e-05,   4.22533748e-02,
               1.17631994e-02]),
      'nit': 28,
      'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH',
      'warnflag': 0})




    len(x0)




    10




    x0 = bt
    fprime(x0)




    array([ 1.91295272,  1.52233617,  1.37382692])




    x0




    array([ 0.        ,  0.        ,  1.26448505,  0.        ,  0.        ,
            1.16071484,  2.2498373 ,  0.46709332,  0.        ,  0.0745338 ])




    n = 5
    p = 3
    amat = np.absolute(npr.normal(0,1,(n,p)))
    bt0 = np.zeros((p,1))
    bt0[:2] = 1
    mu = np.dot(amat,bt0)
    ysq = spst.ncx2.rvs(10,mu**2)
    ls = ncxlosses(10,ysq)
    bt = spo.nnls(amat,np.squeeze(np.sqrt(ysq)))[0]
    x0 = np.array(bt)
    def loss(x0):
        yh = np.dot(amat,x0)
        return sum(np.array([ls[i](yh[i])[0] for i in range(len(yh))]))
    res = bfgssolve(amat,ls,bt,0.0)


    res




    (array([ 1.48059861,  0.99021978,  1.55658785]),
     14.223414168898945,
     {'funcalls': 8,
      'grad': array([ -1.67004136e-06,  -3.28501209e-07,  -6.84309889e-07]),
      'nit': 6,
      'task': 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL',
      'warnflag': 0})




    gd(amat,ls,bt,20)




    (array([ 1.47878171,  0.99154165,  1.55778628]), 14.223420933408791)




    
