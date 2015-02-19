

    import numpy as np
    import numpy.random as npr
    import scipy as sp
    import scipy.stats as spst


    def rician_logpdf(x, nc, resolution = 100):
        pts = 2 * np.pi * np.arange(0, 1.0, 1.0/resolution)
        x_coords = x * np.cos(pts)
        y_coords = x * np.sin(pts)
        exponent = -0.5 * ((x_coords - np.sqrt(nc))**2 + y_coords**2)
        exponent_max = max(exponent)
        logave = np.log(sum(np.exp(exponent - exponent_max))) + exponent_max - np.log(resolution)
        ans = np.log(x) + logave
        return ans


    x = 2.0
    spst.chi.logpdf(x, df = 2), rician_logpdf(x, 0.0, 100)




    (-1.3068528194400546, -1.3068528194400555)




    xs = np.arange(0, 20.0, 0.05)
    nc = 100
    lps = np.array([rician_logpdf(x, nc) for x in xs])
    ps = np.exp(lps)
    nsize = 100000
    smp = np.sqrt(spst.ncx2.rvs(df = 2, nc = nc, size = nsize))


    import matplotlib.pyplot as plt
    %matplotlib inline
    
    hist = plt.hist(smp, bins = 50)
    binwidth = hist[1][1] - hist[1][0]
    plt.plot(xs, ps * nsize * binwidth, lw = 3)
    plt.show()


![png](devRician_files/devRician_4_0.png)



    
