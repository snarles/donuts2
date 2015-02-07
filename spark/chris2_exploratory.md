

    sc




    <pyspark.context.SparkContext at 0x7f1180aaa690>




    f = open('/root/donuts/spark/sparkDonuts.py', 'r')
    exec(f.read())
    f.close()


    def redMean(x, y):
        return (x[0] + y[0], (x[0]*x[1] + y[0]*y[1])/(x[0]+y[0]))
    
    # estimate noncentral chi squared params using method of moments (df known)
    def ncx_mm_est(df, mu, var):
        lambdas = mu * np.arange(0, 1, 0.01)
        sigma2s = (mu - lambdas)/df
        varis = sigma2s*(4*lambdas + 2*df*sigma2s)
        scores = np.absolute(varis-vari)
        ind = np.where(scores == min(scores))[0][0]
        return lambdas[ind], sigma2s[ind]
    
    np.set_printoptions(linewidth = 120, precision = 1)


    parts = 100
    raw = sc.textFile("data/data2", parts)


    arrs = raw.map(str2array).cache()


    smp = arrs.takeSample(False, 100)


    
    #smp[0]


    def align_check(smp):
        ans = []
        if (sum(smp[0,:] != smp[0,0]) + sum(smp[1,:] != smp[1,0]) + sum(smp[2,:] != smp[2,0])) > 0:
            ans.append(smp)
        return ans


    chk_align = arrs.flatMap(align_check)


    from operator import add
    chk_align.map(lambda x : 1).reduce(add)


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-23-886483f9d1e2> in <module>()
          1 from operator import add
    ----> 2 chk_align.map(lambda x : 1).reduce(add)
    

    /root/spark/python/pyspark/rdd.pyc in reduce(self, f)
        716         if vals:
        717             return reduce(f, vals)
    --> 718         raise ValueError("Can not reduce() empty RDD")
        719 
        720     def fold(self, zeroValue, op):


    ValueError: Can not reduce() empty RDD



    arr = smp[0]
    np.shape(arr)




    (14, 32)




    arr = arr[:, arr[3,:]-1]


    #arr[4:,:]


    def varest1(arr):
        subarr = np.array(arr[4:, ]/1000.0, dtype=float)
        combs = np.sum(subarr, axis = 1)
        var1 = np.var(combs, ddof = 1)
        marvars = np.array([np.var(subarr[:,i], ddof=1) for i in range(32)])
        var2 = np.sum(marvars)
        coords = arr[0:3,0]
        key = ints2str(coords)
        #return (key, np.array([var1, var2]))
        #return np.array([var1, var2])
        return (1, np.array([var1, var2]))


    varest1(smp[0])




    ('\\NT', array([ 1846.8228725 ,   276.46831459]))




    smpR = sc.parallelize(smp, 2)


    smpR.map(varest1).reduce(redMean)




    (100, array([ 1360.79829173,   567.80390616]))




    res1 = arrs.map(varest1).reduce(redMean)




    (993600, array([ 38535.56003223,  20968.65408016]))




    mu_var = res1[1]


    mu_var




    array([ 38535.56003223,  20968.65408016])




    corr_factr = mu_var[0]/mu_var[1]


    corr_factr




    1.8377698389660286



## Try out our separate-coil estimation tech


    def cv_varest1(arr):
        corr_factr =1.8377
        cv_inds = [1,2,5]
        tr_inds = list(set(range(10))-set(cv_inds))
        subarr = np.array(arr[4:, ]/1000.0, dtype=float)
        cvarr = subarr[cv_inds, :]
        trarr = subarr[tr_inds, :]
        cvcombs = np.sum(cvarr, axis = 1)
        trcombs = np.sum(trarr, axis = 1)
        # ground truth for CV
        var0 = np.var(cvcombs, ddof = 1)
        trvar1 = np.var(trcombs, ddof = 1)
        marvars = np.array([np.var(trarr[:,i], ddof=1) for i in range(32)])
        trvar2 = corr_factr * np.sum(marvars)
        trvar3 = sum(np.cov(trarr, ddof=1).ravel())
        cve1 = (trvar1-var0)**2
        cve2 = (trvar2-var0)**2
        cve3 = (trvar3-var0)**2
        coords = arr[0:3,0]
        key = ints2str(coords)
        #return (key, np.array([var1, var2]))
        #return np.array([var1, var2])
        return (1, np.array([cve1, cve2, cve3]))


    cv_res01 = arrs.map(cv_varest1).reduce(redMean)


    cv_res01




    (993600, array([  1.8e+10,   1.8e+10,   1.9e+11]))



## Check S0 correlations


    def combs(arr):
        subarr = np.array(arr[4:, ]/1000.0, dtype=float)
        combs = np.sum(subarr, axis = 1)
        return (1, combs)


    mu_s0 = arrs.map(combs).reduce(redMean)
    mu_s0 = mu_s0[1]


    mu_s0




    array([ 90.8,  90.6,  90.3,  89.9,  90.1,  92. ,  91.2,  90.2,  90.4,  90.5])




    def covs(v):
        return (1, np.outer(v[1]-np.mean(v[1]), v[1]-np.mean(v[1])).ravel())


    cov_s0 = arrs.map(combs).map(covs).reduce(redMean)


    cov_s0= np.reshape(cov_s0[1], (10,10))


    np.set_printoptions(precision = 1)
    cov_s0




    array([[ 4927.5,  2278. ,  1412.5,  1175.6,   -32.6, -3264. ,   197.2,  -848.6, -2719.5, -3126.2],
           [ 2278. ,  2964.1,  1334.7,   950.8,  -121.5, -1959.7,   181.3,  -877. , -2195.3, -2555.4],
           [ 1412.5,  1334.7,  2557.4,   584.4,   376.3,  -885.6, -1066.2,  -919.7, -1596.1, -1797.8],
           [ 1175.6,   950.8,   584.4,  2794.8,  -653. ,  -674.9,  -205.9,  -991.7, -1046.1, -1934.1],
           [  -32.6,  -121.5,   376.3,  -653. ,  1976.6,   368.1, -1172.7,  -331.9,  -325.9,   -83.3],
           [-3264. , -1959.7,  -885.6,  -674.9,   368.1,  5842.6, -1709.9,  -804.7,  1790.1,  1298.1],
           [  197.2,   181.3, -1066.2,  -205.9, -1172.7, -1709.9,  5130.2,   329.2,  -472.2, -1210.9],
           [ -848.6,  -877. ,  -919.7,  -991.7,  -331.9,  -804.7,   329.2,  2090.5,   699.7,  1654.2],
           [-2719.5, -2195.3, -1596.1, -1046.1,  -325.9,  1790.1,  -472.2,   699.7,  3440.9,  2424.4],
           [-3126.2, -2555.4, -1797.8, -1934.1,   -83.3,  1298.1, -1210.9,  1654.2,  2424.4,  5331. ]])




    plt.imshow(cov_s0)




    <matplotlib.image.AxesImage at 0x7f116d205d10>




![png](chris2_exploratory_files/chris2_exploratory_34_1.png)



    zip(np.arange(0, 10, 1, dtype=int), np.diag(cov_s0))




    [(0, 4927.503987896338),
     (1, 2964.1314800958639),
     (2, 2557.4105439168052),
     (3, 2794.8249117069658),
     (4, 1976.5849029659867),
     (5, 5842.6137964595291),
     (6, 5130.1964907470028),
     (7, 2090.4848968839692),
     (8, 3440.904646345708),
     (9, 5331.0102087137302)]




    keep_inds = [1,2,3,4,7]


    

## Estimate sigma2 parameter


    def meanArray(x, y):
        z = np.zeros(np.shape(x))
        z[0,:] = x[0,:] + y[0,:]
        for ii in range(1,3):
            z[ii,:] = (x[0,:]*x[ii,:] + y[0,:]*y[ii,:])/(x[0,:] + y[0,:])
        return z
        
    def mu_var(arr):
        #keep_inds = range(10)
        keep_inds = [1,2,3,4,7]
        subarr = np.array(arr[4:, ]/1000.0, dtype=float)
        subarr = subarr[keep_inds, :]
        mus = np.mean(subarr, axis = 0)
        varis = np.var(subarr, axis = 0, ddof=1)
        return np.vstack([np.ones((1,32)), mus, varis])
    
    def estParams(moments):
        return np.array([np.array(ncx_mm_est(2, mu, var)) for (mu, var) in zip(moments[1,:], moments[2,:])])


    moments = arrs.map(mu_var).reduce(meanArray)


    ests = np.array([np.array(ncx_mm_est(2, mu, var)) for (mu, var) in zip(moments[1,:], moments[2,:])])
    vstack([moments[1:,:], ests.T]).T
    # mean should be lambda+2*sigma2, variance should be sigma2(4*lambda + 4*sigma2)




    array([[   1.5,    6.3,    0. ,    0.7],
           [   4.7,   21.9,    0. ,    2.3],
           [   4.2,   18.5,    0. ,    2.1],
           [   2.3,    6.4,    0. ,    1.2],
           [   1.5,    3.4,    0. ,    0.7],
           [   5.3,   37.6,    0. ,    2.7],
           [   3.4,   14.9,    0. ,    1.7],
           [   2.8,    7.4,    0. ,    1.4],
           [   2.3,    7. ,    0. ,    1.1],
           [   2.7,    8.2,    0. ,    1.4],
           [   3.9,   16.3,    0. ,    2. ],
           [   1.3,    2.1,    0. ,    0.6],
           [   2.2,    8.8,    0. ,    1.1],
           [   3.6,   13.3,    0. ,    1.8],
           [   3.9,   11.1,    0. ,    1.9],
           [   1.5,    4.8,    0. ,    0.7],
           [   1.7,    8.4,    0. ,    0.8],
           [   4.5,   58.3,    0. ,    2.3],
           [   4.3,   28. ,    0. ,    2.1],
           [   2.2,    8.8,    0. ,    1.1],
           [   3. ,  199.5,    0. ,    1.5],
           [   4.1,   57.3,    0. ,    2. ],
           [   2.2,   15.2,    0. ,    1.1],
           [   1.7,    6.7,    0. ,    0.9],
           [   1.5,    6.5,    0. ,    0.7],
           [   1.9,   13.3,    0. ,    1. ],
           [   3.2,   33.9,    0. ,    1.6],
           [   2.7,  133.6,    0. ,    1.3],
           [   2. ,   10.3,    0. ,    1. ],
           [   4. ,   33.1,    0. ,    2. ],
           [   3.5,   39.8,    0. ,    1.7],
           [   1. ,    3.3,    0. ,    0.5]])




    smp = arrs.map(mu_var).takeSample(False, 10)


    np.hstack([smp[4].T,estParams(smp[4])])




    array([[  1.0e+00,   2.3e-02,   5.5e-04,   0.0e+00,   1.2e-02],
           [  1.0e+00,   2.2e-02,   4.7e-04,   0.0e+00,   1.1e-02],
           [  1.0e+00,   7.1e-03,   4.5e-05,   0.0e+00,   3.6e-03],
           [  1.0e+00,   1.5e-02,   2.9e-04,   0.0e+00,   7.3e-03],
           [  1.0e+00,   2.6e-02,   9.1e-04,   0.0e+00,   1.3e-02],
           [  1.0e+00,   3.0e-02,   4.3e-04,   0.0e+00,   1.5e-02],
           [  1.0e+00,   2.6e-02,   8.8e-04,   0.0e+00,   1.3e-02],
           [  1.0e+00,   3.0e-02,   1.4e-03,   0.0e+00,   1.5e-02],
           [  1.0e+00,   2.4e-02,   1.2e-03,   0.0e+00,   1.2e-02],
           [  1.0e+00,   3.1e-02,   2.2e-03,   0.0e+00,   1.6e-02],
           [  1.0e+00,   1.6e-02,   6.7e-04,   0.0e+00,   8.0e-03],
           [  1.0e+00,   1.6e-02,   6.8e-04,   0.0e+00,   7.8e-03],
           [  1.0e+00,   1.3e-02,   2.1e-04,   0.0e+00,   6.7e-03],
           [  1.0e+00,   1.1e-02,   3.2e-04,   0.0e+00,   5.5e-03],
           [  1.0e+00,   1.4e-02,   4.2e-04,   0.0e+00,   7.2e-03],
           [  1.0e+00,   2.6e-02,   7.7e-04,   0.0e+00,   1.3e-02],
           [  1.0e+00,   3.0e-02,   2.8e-03,   0.0e+00,   1.5e-02],
           [  1.0e+00,   5.7e-03,   3.5e-05,   0.0e+00,   2.9e-03],
           [  1.0e+00,   1.1e-02,   1.8e-04,   0.0e+00,   5.3e-03],
           [  1.0e+00,   1.6e-02,   7.7e-05,   0.0e+00,   7.8e-03],
           [  1.0e+00,   3.5e-02,   2.5e-03,   0.0e+00,   1.7e-02],
           [  1.0e+00,   2.4e-02,   9.5e-04,   0.0e+00,   1.2e-02],
           [  1.0e+00,   1.7e-02,   5.8e-04,   0.0e+00,   8.3e-03],
           [  1.0e+00,   1.1e-02,   1.6e-04,   0.0e+00,   5.6e-03],
           [  1.0e+00,   1.3e-02,   3.2e-04,   0.0e+00,   6.3e-03],
           [  1.0e+00,   2.4e-02,   1.6e-03,   0.0e+00,   1.2e-02],
           [  1.0e+00,   2.2e-02,   8.8e-04,   0.0e+00,   1.1e-02],
           [  1.0e+00,   1.7e-02,   1.7e-04,   0.0e+00,   8.7e-03],
           [  1.0e+00,   1.9e-02,   8.2e-04,   0.0e+00,   9.7e-03],
           [  1.0e+00,   2.5e-02,   1.2e-03,   0.0e+00,   1.3e-02],
           [  1.0e+00,   2.7e-02,   8.0e-04,   0.0e+00,   1.3e-02],
           [  1.0e+00,   2.6e-02,   1.2e-03,   0.0e+00,   1.3e-02]])




    mu = 10; sigma = 4.0
    lala = rvs_ncx2(2, mu, 1000, sigma)
    #np.mean(lala), mu**2 + 64*sigma**2, np.var(lala), sigma**2*(4*mu**2 + 128*sigma**2)
    sigma2hat = np.var(lala, ddof = 1)/np.mean(lala) * .5
    lmbdahat = np.mean(lala) - 2 * sigma2hat
    sigma2hat, lmbdahat




    (26.70000877279233, 73.309466705543471)



$$
\frac{Var}{Mean} = \frac{\sigma^2 (4\lambda + 2 df \sigma^2)}{\lambda + df
\sigma^2}\approx 2\sigma^2
$$


    


    np.var(lala)/np.mean(lala), 2*sigma**2




    (34.854593819541869, 32.0)




    sigma2 = moments[1]/moments[0] * 0.5
    lmb = moments[0] - 2*sigma2


    lmb




    52.641705481189348




    
