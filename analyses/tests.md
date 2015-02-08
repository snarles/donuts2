
# Development of nosetests


    reload(ncx)




    <module 'donuts.deconv.ncx' from 'donuts/deconv/ncx.py'>




    import os
    import numpy as np
    import numpy.random as npr
    import numpy.linalg as nla
    import scipy.stats as spst
    import scipy as sp
    import scipy.optimize as spo
    import numpy.testing as npt
    import scipy.spatial.distance as spd
    os.chdir("..")
    import donuts.deconv.utils as du
    import donuts.deconv.ncx as ncx
    import donuts.deconv.splines as spl


    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)


## Test utils


    def test_scalarize():
        assert type(du.scalarize(5)) == np.int64
        assert len(du.scalarize(np.ones(5))) ==5
        assert len(du.scalarize(np.ones((5,2)))) ==5
    
    def test_column():
        assert np.shape(du.column(5))[1]==1
        assert np.shape(du.column([5,4]))[1]==1
        assert np.shape(du.column(np.ones((4,3))))==(12,1)
        
    def test_rank_simple():
        temp = npr.normal(0,1,10)
        x = temp[npr.randint(0,10,100)]
        o = du.rank_simple(x)
        assert x[o[0]] == min(x)
        assert x[o[-1]] == max(x)
        xs = np.array(x)[o]
        i = du.rank_simple(o)
        npt.assert_almost_equal(x,xs[i])
    
    def test_fullfact():
        a = du.fullfact([3,3,3])
        npt.assert_almost_equal(np.shape(a)[0], 27)
    
    def test_inds_fullfact():
        levels = npr.randint(2,5,4).tolist()
        a = du.fullfact(levels)
        inds1 = du.inds_fullfact(levels,[0,1],[1,1])
        inds2 = np.where(np.logical_and(a[:,0]==1, a[:,1]==1))[0]
        npt.assert_almost_equal(inds1,inds2)    
    
    def test_ordered_partitions():
        n = npr.randint(4,10)
        k = npr.randint(2,n)
        ans = du.ordered_partitions(n,k)
        rowsums = [sum(a) for a in ans]
        npt.assert_almost_equal(rowsums,[n]*np.shape(ans)[0])
        toint = np.array([sum(a * np.power(n,k-np.arange(0,k,1.0))) for a in ans])
        npt.assert_almost_equal(du.rank_simple(toint),range(np.shape(ans)[0]))
        
    def test_normalize_rows():
        a = du.normalize_rows(np.random.normal(0,1,(10000,3)))
        npt.assert_almost_equal(sum(a[1,]**2),1)
    
    def test_ste_tan_kappa():
        def subroutine(kappa):
            vs = du.normalize_rows(npr.normal(0,1,(10,3)))
            bvecs = np.sqrt(kappa) * du.normalize_rows(npr.normal(0,1,(1000,3)))
            amat = du.ste_tan_kappa(vs,bvecs)
            mus = np.array([np.mean(a) for a in amat.T])
            stds = np.array([np.std(a) for a in amat.T])
            assert np.std(mus) < 0.1
            assert np.std(stds) < 0.1
            return np.mean(mus)
        kappas = np.arange(0.1,3,0.1)
        muss = np.array([subroutine(kappa) for kappa in kappas])
        npt.assert_almost_equal(du.rank_simple(-muss),range(len(muss)))
        
    def test_simulate_signal_kappa():
        # part I. test for approximate gaussianity
        for ii in range(10):
            true_kappa = abs(npr.normal(2,1,1))[0]
            bvecs = np.sqrt(true_kappa)*du.geosphere(4)
            true_pos = du.normalize_rows(npr.normal(0,1,(3,3)))
            true_w = np.array([1.,1.,1.]).reshape((-1,1))
    
            count = 0.0
            sigma = 1e-2
            if true_kappa < 1.0:
                sigma = 1e-3
            flag = True
            df = 4
            y0, y1 = du.simulate_signal_kappa(true_pos,true_w,bvecs,sigma,df)
            pred_mu = ncx.mean_ncx(df,y0,sigma)
            mu = 0.0 * np.array(y0)
            vr = 0.0 * np.array(y0)
            skw = 0.0 * np.array(y0)
            while flag:
                count = count + 1.0
                temp, y1 = du.simulate_signal_kappa(true_pos,true_w,bvecs,sigma)
                mu = (count/(count+1))* mu + (1/(count+1))*(y1-y0)
                vr = (count/(count+1))* vr + (1/(count+1))*(y1-y0)**2
                skw = (count/(count+1))* skw + (1/(count+1))*(y1-y0)**3
                if (count >= 10) and (max(abs(mu)) < 1e-2) and (max(abs(vr - sigma**2)) < 1e-4) and \
                (max(abs(skw)) < 1e-5) and (max(abs(mu+y0-pred_mu)*y0) < 1e-2):
                    flag=False
                assert count < 100
            if count > 70:
                print("test_simulate_signal_kappa (part I) " + str(count) + " / 70 trials needed (kappa="+str(true_kappa)+")")
        # part II. test for large sigma
        for ii in range(10):
            true_kappa = abs(npr.normal(2,1,1))[0]
            bvecs = np.sqrt(true_kappa)*du.geosphere(4)
            true_pos = du.normalize_rows(npr.normal(0,1,(3,3)))
            true_w = np.array([1.,1.,1.]).reshape((-1,1))
    
            count = 0.0
            sigma = 1000*np.exp(abs(npr.normal(5,1,1))[0])
            flag = True
            df = 10
            y0, y1 = du.simulate_signal_kappa(true_pos,true_w,bvecs,sigma,df)    
            mu = 0.0 * np.array(y0)            
            while flag:
                count = count + 1.0
                temp, y1 = du.simulate_signal_kappa(true_pos,true_w,bvecs,sigma)
                mu = (count/(count+1))* mu + (1/(count+1))*(y1)
                if (count >= 10) and (np.std(mu)/np.mean(mu) < 1e-1):
                    flag=False
                assert count < 70
            if count > 40:
                print("test_simulate_signal_kappa (part II) " + str(count) + " / 40 trials needed (kappa="+str(true_kappa)+")")
    
    def test_rand_ortho():
        a=du.rand_ortho(3)
        npt.assert_almost_equal(np.dot(a.T,a),np.eye(3))
    
    def test_arcdist():
        # check the distances between 3 orthogonal vectors
        a = du.rand_ortho(3)
        npt.assert_almost_equal(du.arcdist(a,a),np.pi/2 * np.ones((3,3)) - np.pi/2 * np.eye(3))
        # check the distances between points on a random great circle
        thetas = npr.uniform(0,np.pi/2,10)
        v0 = np.dot(np.vstack([np.cos(thetas),np.sin(thetas),0.0*thetas]).T,a)
        def wrap(x0):
            x = np.absolute(x0)
            x[x > np.pi/2] = x[x > np.pi/2] - np.pi/2
            return x
        npt.assert_almost_equal(du.arcdist(v0,v0),wrap(du.column(thetas) - du.column(thetas).T))
        
    def test_randvecsgap():
        a = du.randvecsgap(20,.1)
        assert np.sum(du.arcdist(a,a) < .1)==20
        a = du.randvecsgap(3,.5)
        assert np.sum(du.arcdist(a,a) < .5)==3
    
    def test_arc_emd():
        # NOTE: emd is actually not sufficiently precise to 7 decimal places
        #  hence the fudge factor
        fudge = 1e-1
        for ii in range(10):
            eps = npr.uniform(0.0,0.1)
            # an orthogonal set of vectors
            v0 = du.rand_ortho(3)
            w0 = du.column(np.sort(npr.exponential(1,size=3)))
            w0 = w0/sum(w0)
            # test reflective symmetry and weight renormalization
            npt.assert_almost_equal(du.arc_emd(v0,w0,-v0,2*w0),0.0)
            # test invariance under splitting
            v0x2 = np.vstack([v0,v0])
            w0x2 = np.vstack([w0,w0])*.5
            npt.assert_almost_equal(du.arc_emd(v0,w0,v0x2,w0x2),0.0)
            # test EMD for fixed directions but different weights
            w0p = w0 + du.column([eps,0,-eps])
            npt.assert_almost_equal(fudge*du.arc_emd(v0,w0,v0,w0p),fudge*eps*np.pi/2)
            # test EMD for fixed weights but perturbed directions
            v0p = du.normalize_rows(v0 + eps * du.randvecsgap(3,0.0))
            npt.assert_almost_equal(fudge*du.arc_emd(v0,w0,v0p,w0),fudge*sum(np.diag(du.arcdist(v0,v0p)*np.squeeze(w0))))
    
    def test_geosphere():
        ksph = npr.randint(2,8)
        bvecs = du.geosphere(ksph)
        pp = np.shape(bvecs)[0]
        # a random positive function on the sphere
        kdir = 3
        dirs = du.randvecsgap(kdir,0.0)
        cfs = npr.normal(0,1,kdir)
        def sphpoly(v): # input a (3,) array
            dps = np.squeeze(np.dot(dirs,v))
            return np.exp(sum(cfs*dps**8))
        # test rotational invariance of bvecs compared to random design points
        record = np.zeros(10)
        for iii in range(10):
            def subroutine1():
                oo = du.rand_ortho(3)
                bvecs2 = np.dot(bvecs,oo)
                y1 = np.array([sphpoly(v) for v in bvecs2])
                return np.mean(y1)
            def subroutine0():
                bvecs2 = du.randvecsgap(pp,0.0)
                y1 = np.array([sphpoly(v) for v in bvecs2])
                return np.mean(y1)
            mus1 = np.array([subroutine1() for ii in range(20)])
            mus0 = np.array([subroutine0() for ii in range(20)])
            record[iii] = (np.std(mus1)/np.mean(mus1) < .3*np.std(mus0)/np.mean(mus0))
        assert sum(record) > 5
        if sum(record) < 10:
            print("test_geosphere(" + str(ksph) + "), success: "+str(sum(record))+"/10")


    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    y1 = [sphpoly(v) for v in bvecs]
    zz = np.squeeze(y1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(zz*bvecs[:,0],zz*bvecs[:,1],zz*bvecs[:,2])
    plt.show()

# Test splines


    def test_b4sd():
        x = np.arange(-2,6,0.01)
        y = spl.bs4d(x)
        def f(x):
            return spl.bs4d(x)[0]
        npt.assert_almost_equal(y[1],ncx.numderiv(f,x,1e-8),decimal = 5)
        npt.assert_almost_equal(y[2],ncx.numderiv2(f,x,1e-4),decimal = 2)
        return
    
    def test_splinemat():
        a = spl.splinemat(npr.randint(5,10))
        npt.assert_almost_equal(a[0,0:3],np.array([1,4,1]))
        npt.assert_almost_equal(a[-1,-3:],np.array([1,4,1]))
        return
    
    def test_splinemat2():
        a = spl.splinemat2(npr.randint(5,10))
        npt.assert_almost_equal(a[0,0:3],np.array([6,-12,6]))
        npt.assert_almost_equal(a[-1,-3:],np.array([6,-12,6]))
        return
    
    def test_splinecoef():
        k = npr.randint(3,10)
        v = npr.normal(0,1,k)
        a = spl.splinemat(k)
        b = spl.splinecoef(k)
        bt = np.dot(b,v)
        npt.assert_almost_equal(np.dot(a,bt),v)
        return
    
    def test_genspline():
        k = npr.randint(10,20)
        v = npr.normal(0,1,k)
        a = spl.splinemat(k)
        b = spl.splinecoef(k)
        bt = np.dot(b,v)
        f = spl.genspline(bt,1,0)
        npt.assert_almost_equal([f(x)[0][0] for x in np.arange(1,k,1)],v[1:])
        return
    
    def test_autospline():
        # random cosine function
        kc = 10
        coefs = npr.normal(0,1,kc)
        def f(x):
            exmat = np.dot(du.column(x),du.column(range(kc)).T)
            return np.dot(np.cos(exmat),coefs)
        lb = npr.normal(0,1)
        ub = lb + npr.exponential(2)
        x = np.arange(lb,ub,(ub-lb)/npr.randint(30,70))
        y = f(x)
        f2 = spl.autospline(x,y)
        xgrid = np.arange(x[5],x[-5],(ub-lb)/1000)
        ygrid = f(xgrid)
        yspl = np.array([f2(xx)[0] for xx in xgrid])
        temp = np.cov(ygrid.T,yspl.T)
        npt.assert_almost_equal(temp[1,0]/temp[0,0],1.0,decimal=2)
        #plt.scatter(xgrid,ygrid)
        #plt.scatter(xgrid,yspl,color="red")
        #plt.show()
        return
    
    def test_convspline():
        kc = 10
        cfsa = .1*npr.uniform(-3,3,kc)
        cfsb = npr.uniform(-1,1,kc)
        def lsef(x): # scalar argument
            return np.log(sum(np.exp(cfsa * x + cfsb)))
        lb = npr.uniform(-10,-5)
        ub = npr.uniform(5,10)
        x = np.arange(lb,ub,(ub-lb)/npr.randint(30,70))
        y = np.array([lsef(xx) for xx in x])
        f = spl.convspline(x,y)
        y2 = np.array([f(xx)[0] for xx in x])
        temp = np.cov(y[1:-2].T,y2[1:-2].T)
        npt.assert_almost_equal(temp[1,0]/temp[0,0],1.0,decimal=2)
        #plt.scatter(x,y)
        #plt.scatter(x,y2,color="red")
        #plt.show()
        for ii in range(100):
            p1 = npr.normal(0,1000)
            p2 = npr.normal(0,1000)
            assert f((p1+p2)/2.0)[0] < (f(p1)[0]+f(p2)[0])/2.0
        return


    kc = 10
    cfsa = .1*npr.uniform(-3,3,kc)
    cfsb = npr.uniform(-1,1,kc)
    def lsef(x): # scalar argument
        return np.log(sum(np.exp(cfsa * x + cfsb)))
    lb = npr.uniform(-10,-5)
    ub = npr.uniform(5,10)
    x = np.arange(lb,ub,(ub-lb)/npr.randint(30,70))
    y = np.array([lsef(xx) for xx in x])
    f = spl.convspline(x,y)
    y2 = np.array([f(xx)[0] for xx in x])
    temp = np.cov(y[1:-2].T,y2[1:-2].T)
    npt.assert_almost_equal(temp[1,0]/temp[0,0],1.0,decimal=2)
    #plt.scatter(x,y)
    #plt.scatter(x,y2,color="red")
    #plt.show()
    for ii in range(100):
        p1 = npr.normal(0,1000)
        p2 = npr.normal(0,1000)
        assert f((p1+p2)/2.0)[0] < (f(p1)[0]+f(p2)[0])/2.0

## Test NCX


    def test_numderiv():
        def f(x):
            return x**2
        x = np.arange(-5,5,0.1)
        npt.assert_almost_equal(2*x,ncx.numderiv(f,x,1e-3),decimal=3)
        return
    def test_numderiv2():
        def f(x):
            return x**2
        x = np.arange(-5,5,0.1)
        npt.assert_almost_equal(2+0*x,ncx.numderiv2(f,x,1e-3),decimal=3)
        return
    
    def test_mean_ncx():
        for ii in range(5):
            df = npr.randint(2,20)
            mu = npr.uniform(0,10)
            x = spst.ncx2.rvs(df,mu**2,size=1e6)
            npt.assert_almost_equal(mu*np.mean(np.sqrt(x)),mu*ncx.mean_ncx(df,mu),decimal=1)
        return
    
    def test_ncxloss_gauss():
        # demonstrate approximation to true loss function
        def subroutine(df,mu0):
            x = spst.ncx2.rvs(df,mu0**2)
            mus = np.arange(mu0*0.1,2*mu0,mu0*0.1)
            f0 = ncx.ncxloss_true(x,df)
            f = ncx.ncxloss_gauss(x,df)
            y0= f0(mus)[0]
            y = f(mus)[0]
            return spst.pearsonr(y,y0)[0]
        # moderate accuracy for small arguments
        reps = 50
        pars = zip(npr.randint(2,20,size=reps),npr.uniform(1,10,size=reps))
        rs= np.array([subroutine(df,mu0) for (df,mu0) in pars])
        assert sum(rs < .5) < 10
        # high accuracy for large arguments
        df = npr.randint(5,10)
        x = npr.uniform(70,100)
        mus = np.arange(30,70,1.0)
        f0 = ncx.ncxloss_true(x,df)
        f = ncx.ncxloss_gauss(x,df)
        y0= f0(mus)[0]
        y = f(mus)[0]
        npt.assert_almost_equal(spst.pearsonr(y,y0)[0],1,decimal=3)
        # test derivatives
        def fval(x):
            return f(x)[0]
        npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=3)
        # demonstrate asymptotic consistency
        df = npr.randint(5,20)
        mu0 = npr.uniform(20,50)
        x = spst.ncx2.rvs(df,mu0**2,size=1e4)
        ls = [ncx.ncxloss_gauss(xx,df) for xx in x]
        def likelihood(mu):
            return sum(np.array([ll(mu)[0] for ll in ls]))
        lk0 = likelihood(mu0)
        assert lk0 < likelihood(mu0 * .5)
        assert lk0 < likelihood(mu0 * 1.01)
        assert lk0 < likelihood(mu0 * 0.99)
        assert lk0 < likelihood(mu0 * 2)
        assert lk0 < likelihood(mu0 * npr.uniform(1e-2,100))
        return
    
    def test_ncxloss_mean():
            # demonstrate asymptotic consistency
        df = npr.randint(5,20)
        mu0 = npr.uniform(20,50)
        x = spst.ncx2.rvs(df,mu0**2,size=1e5)
        ls = [ncx.ncxloss_mean(xx,df) for xx in x]
        def likelihood(mu):
            return sum(np.array([ll(mu)[0] for ll in ls]))
        lk0 = likelihood(mu0)
        assert lk0 < likelihood(mu0 * 1.1)
        assert lk0 < likelihood(mu0 * 0.9)
        # test derivatives
        f = ls[0]
        mus = np.arange(0.1,2.0,0.1)*mu0
        def fval(x):
            return f(x)[0]
        npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=3)
        return
    
    def test_ncxloss_true():
        # demonstrate asymptotic consistency
        df = npr.randint(2,4)
        mu0 = npr.uniform(0,2)
        x = spst.ncx2.rvs(df,mu0**2,size=1e3)
        ls = [ncx.ncxloss_true(xx,df) for xx in x]
        def likelihood(mu):
            return sum(np.array([ll(mu)[0] for ll in ls]))
        lk0 = likelihood(mu0)
        assert lk0 < likelihood(mu0 * .5)
        assert lk0 < likelihood(mu0 * 2)
        return
    
    def test_rvs_ncx2():
        for ii in range(10):
            df = npr.randint(1,100)
            sigma = npr.uniform(.1,3)
            mu = npr.uniform(0,10)
            x = ncx.rvs_ncx2(df,mu,1000000,sigma)
            #x = sigma*spst.ncx2.rvs(df,(mu/sigma)**2,size = 100000)
            npt.assert_almost_equal(np.mean(x)/(mu**2 + (sigma**2)*df),1,decimal=2)
            npt.assert_almost_equal(np.var(x)/(4*(sigma**2)*mu**2 + 2*(sigma**4)*df),1,decimal=2)
        return
    
        


    reload(ncx)




    <module 'donuts.deconv.ncx' from 'donuts/deconv/ncx.py'>




    for ii in range(10):
        df = npr.randint(1,100)
        sigma = npr.uniform(.1,3)
        mu = npr.uniform(0,10)
        x = ncx.rvs_ncx2(df,mu,1000000,sigma)
        #x = sigma*spst.ncx2.rvs(df,(mu/sigma)**2,size = 100000)
        npt.assert_almost_equal(np.mean(x)/(mu**2 + (sigma**2)*df),1,decimal=2)
        npt.assert_almost_equal(np.var(x)/(4*(sigma**2)*mu**2 + 2*(sigma**4)*df),1,decimal=2)


    spst.ncx2.rvs(10,1e-2)




    5.7990307938018155




    spst.chi2.rvs(10)




    4.881512673379833




    def test_ncxloss_gauss():
        # demonstrate approximation to true loss function
        def subroutine(df,mu0):
            x = spst.ncx2.rvs(df,mu0**2)
            mus = np.arange(mu0*0.1,2*mu0,mu0*0.1)
            f0 = ncx.ncxloss_true(x,df)
            f = ncx.ncxloss_gauss(x,df)
            y0= f0(mus)[0]
            y = f(mus)[0]
            return spst.pearsonr(y,y0)[0]
        # moderate accuracy for small arguments
        reps = 50
        pars = zip(npr.randint(2,20,size=reps),npr.uniform(1,10,size=reps))
        rs= np.array([subroutine(df,mu0) for (df,mu0) in pars])
        assert sum(rs < .5) < 10
        # high accuracy for large arguments
        df = npr.randint(5,10)
        x = npr.uniform(70,100)
        mus = np.arange(30,70,1.0)
        f0 = ncx.ncxloss_true(x,df)
        f = ncx.ncxloss_gauss(x,df)
        y0= f0(mus)[0]
        y = f(mus)[0]
        npt.assert_almost_equal(spst.pearsonr(y,y0)[0],1,decimal=3)
        # test derivatives
        def fval(x):
            return f(x)[0]
        npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=3)
        # demonstrate asymptotic consistency
        df = npr.randint(5,20)
        mu0 = npr.uniform(20,50)
        sigma = npr.uniform(.5,2.0)
        x = ncx.rvs_ncx2(df,mu0,1e4,sigma)
        ls = [ncx.ncxloss_gauss(xx,df,sigma) for xx in x]
        def likelihood(mu):
            return sum(np.array([ll(mu)[0] for ll in ls]))
        lk0 = likelihood(mu0)
        assert lk0 < likelihood(mu0 * .5)
        assert lk0 < likelihood(mu0 * 1.01)
        assert lk0 < likelihood(mu0 * 0.99)
        assert lk0 < likelihood(mu0 * 2)
        assert lk0 < likelihood(mu0 * npr.uniform(1e-2,100))
        return



    df = npr.randint(5,20)
    mu0 = npr.uniform(20,50)
    sigma = npr.uniform(.5,2.0)
    x = ncx.rvs_ncx2(df,mu0,1e4,sigma)
    ls = [ncx.ncxloss_gauss(xx,df,sigma) for xx in x]
    def likelihood(mu):
        return sum(np.array([ll(mu)[0] for ll in ls]))
    lk0 = likelihood(mu0)
    assert lk0 < likelihood(mu0 * .5)
    assert lk0 < likelihood(mu0 * 1.01)
    assert lk0 < likelihood(mu0 * 0.99)
    assert lk0 < likelihood(mu0 * 2)
    assert lk0 < likelihood(mu0 * npr.uniform(1e-2,100))


    def test_ncxloss_gauss():
        # demonstrate approximation to true loss function
        def subroutine(df,mu0):
            x = spst.ncx2.rvs(df,mu0**2)
            mus = np.arange(mu0*0.1,2*mu0,mu0*0.1)
            f0 = ncx.ncxloss_true(x,df)
            f = ncx.ncxloss_gauss(x,df)
            y0= f0(mus)[0]
            y = f(mus)[0]
            return spst.pearsonr(y,y0)[0]
        # moderate accuracy for small arguments
        reps = 50
        pars = zip(npr.randint(2,20,size=reps),npr.uniform(1,10,size=reps))
        rs= np.array([subroutine(df,mu0) for (df,mu0) in pars])
        assert sum(rs < .5) < 10
        # high accuracy for large arguments
        df = npr.randint(5,10)
        x = npr.uniform(70,100)
        mus = np.arange(30,70,1.0)
        f0 = ncx.ncxloss_true(x,df)
        f = ncx.ncxloss_gauss(x,df)
        y0= f0(mus)[0]
        y = f(mus)[0]
        npt.assert_almost_equal(spst.pearsonr(y,y0)[0],1,decimal=3)
        # test derivatives
        def fval(x):
            return f(x)[0]
        npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=3)
        # demonstrate asymptotic consistency
        df = npr.randint(5,20)
        mu0 = npr.uniform(20,50)
        x = spst.ncx2.rvs(df,mu0**2,size=1e4)
        ls = [ncx.ncxloss_gauss(xx,df) for xx in x]
        def likelihood(mu):
            return sum(np.array([ll(mu)[0] for ll in ls]))
        lk0 = likelihood(mu0)
        assert lk0 < likelihood(mu0 * .5)
        assert lk0 < likelihood(mu0 * 1.01)
        assert lk0 < likelihood(mu0 * 0.99)
        assert lk0 < likelihood(mu0 * 2)
        assert lk0 < likelihood(mu0 * npr.uniform(1e-2,100))
        return


    # test sigma parameter
    
    df = npr.randint(5,20)
    mu0 = npr.uniform(20,50)
    x1 = ncx.rvs_ncx2(df,mu0,1.0)
    x2 = ncx.rvs_ncx2(df,mu0,1.0)
    ls1 = ncx.ncxloss_gauss(x1,df)
    ls2 = ncx.ncxloss_gauss(x2,df)
    mus = np.arange(x1/2,2*x1,x1/10)
    lk1 = ls1(mus)[0]
    lk2 = ls2(mus)[0]
    
    sigma = 0.1
    mu0s = mu0 * sigma
    muss = mus * sigma
    x1s = x1*(sigma**2)
    x2s = x2*(sigma**2)
    ls1s = ncx.ncxloss_gauss(x1s,df,sigma)
    ls2s = ncx.ncxloss_gauss(x2s,df,sigma)
    lk1s = ls1s(muss)[0]
    lk2s = ls2s(muss)[0]
    
    import matplotlib.pyplot as plt


    
    plt.scatter(lk1,lk1s); plt.show()


    plt.scatter(lk2,lk2s); plt.show()


    plt.scatter(10*lk1/lk2,10*lk1s/lk2s); plt.show()


    ncx.ncxloss_gauss(5.0,10.0,1.0)(2.0)




    (3.8356980024327276, 0.7222222222222223)




    ncx.ncxloss_gauss(2.0,4.0,0.1)(2.5)




    (36.918178271777187, -2800.5015529253747)




    reload(ncx)
    # test derivatives
    df = npr.randint(5,10)
    x = npr.uniform(70,100)
    mus = np.arange(30,70,1.0)
    sigma = npr.uniform(.5,2.0)
    f = ncx.ncxloss_gauss(x,df,sigma)
    def fval(x):
        return f(x)[0]
    npt.assert_almost_equal(f(mus)[1],ncx.numderiv(fval,mus,1e-3),decimal=3)
    npt.assert_almost_equal(f(mus)[2],ncx.numderiv2(fval,mus,1e-3),decimal=3)



    f(2.0)




    <function donuts.deconv.ncx.ff>




    