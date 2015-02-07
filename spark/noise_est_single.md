
## Looking at noise in real data


    import cvxopt as cvx
    import numpy as np
    import numpy.random as npr
    import scipy as sp
    import scipy.optimize as spo
    import matplotlib.pyplot as plt
    import subprocess


    import pyspark.context

## Misc dependences


    # Wensheng Wang
    # http://code.activestate.com/recipes/366178-a-fast-prime-number-list-generator/
    def primes(n): 
        if n==2: return [2]
        elif n<2: return []
        s=range(3,n+1,2)
        mroot = n ** 0.5
        half=(n+1)/2-1
        i=0
        m=3
        while m <= mroot:
            if s[i]:
                j=(m*m-3)/2
                s[j]=0
                while j<half:
                    s[j]=0
                    j+=m
            i=i+1
            m=2*i+3
        return [2]+[x for x in s if x]
    
    pms = primes(10000000)


    def coords2key(coords):
        return coords[2] + coords[1]*76 + coords[0]*106*76
    
    def key2coords(key):
        c3 = key % 76
        c2 = ((key-c3)/76) % 106
        c1 = (key - c3 - 76*c2)/(106*76)
        return np.array([c1, c2, c3])


    print(pms[347354],pms[234287])

    (4982273, 3260597)



    def modexp ( g, u, p ):
       """computes s = (g ^ u) mod p
          args are base, exponent, modulus
          (see Bruce Schneier's book, _Applied Cryptography_ p. 244)
          http://stackoverflow.com/questions/5486204/fast-modulo-calculations-in-python-and-ruby"""
       s = 1
       while u != 0:
          if u & 1:
             s = (s * g)%p
          u >>= 1
          g = (g * g)%p;
       return s
    
    def lamehash(key):
        return modexp(4982273, key, 3260597)

## Crunch time


    braindata = np.loadtxt('/root/all_b1000_1_data.csv', delimiter=',', usecols=range(15))


    nvox = np.shape(braindata)[0]


    means = np.array([np.mean(braindata[ii,5:15]) for ii in range(nvox)])


    plt.hist(means)
    plt.show()


![png](noise_est_single_files/noise_est_single_12_0.png)



    braindatalist = [(ii, braindata[ii,0:15]) for ii in range(nvox) if means[ii] > 500]
    len(braindatalist)




    177672




    partitions = 36
    nsubvox = len(braindatalist)
    brain_rdd = sc.parallelize(braindatalist[1:nsubvox], partitions)


    max(braindata[:,2]), max(braindata[:,3]), max(braindata[:,4])




    (81.0, 106.0, 76.0)




    def process_noise_rdd(tup):
        key = tup[0]
        vec = tup[1]
        coords = vec[2:5]
        cv_ind = lamehash(key) % 10
        y = vec[5:15]
        return (key, {'y' : y, 'coords' : coords, 'cv_ind' : cv_ind})


    from operator import add
    sum_rdd = processed_rdd.map(lambda x : x[1]['y']).reduce(add)


    tot_mean = sum_rdd/nsubvox
    tot_mean




    array([ 1020.61,  1020.46,  1020.52,  1020.82,  1020.58,  1020.38,  1019.72,  1019.99,  1020.07,  1020.49])




    def demean_rdd(tup): # demean by row
        dc = tup[1]
        dc['y'] = dc['y'] - np.mean(dc['y'])
        return tup
    
    def calc_cov(tup):
        key = tup[0]
        dc = tup[1]
        y = dc['y']
        cc = np.outer(y, y)
        return (key, cc)


    xtx_rdd = processed_rdd.map(demean_rdd).map(calc_cov).reduce(add)


    np.set_printoptions(precision=1)
    numpy.set_printoptions(linewidth=120)
    xtx_rdd[1]/max(xtx_rdd[1].ravel())




    array([[  5.3e-01,   7.1e-01,   8.6e-02,  -4.9e-01,  -7.0e-03,   1.6e-02,   1.3e-01,  -1.5e-01,  -7.3e-01,  -1.0e-01],
           [  7.1e-01,   9.6e-01,   1.2e-01,  -6.7e-01,  -9.4e-03,   2.2e-02,   1.8e-01,  -2.0e-01,  -9.8e-01,  -1.3e-01],
           [  8.6e-02,   1.2e-01,   1.4e-02,  -8.0e-02,  -1.1e-03,   2.6e-03,   2.2e-02,  -2.4e-02,  -1.2e-01,  -1.6e-02],
           [ -4.9e-01,  -6.7e-01,  -8.0e-02,   4.6e-01,   6.5e-03,  -1.5e-02,  -1.2e-01,   1.4e-01,   6.8e-01,   9.3e-02],
           [ -7.0e-03,  -9.4e-03,  -1.1e-03,   6.5e-03,   9.2e-05,  -2.1e-04,  -1.7e-03,   1.9e-03,   9.6e-03,   1.3e-03],
           [  1.6e-02,   2.2e-02,   2.6e-03,  -1.5e-02,  -2.1e-04,   5.0e-04,   4.1e-03,  -4.5e-03,  -2.2e-02,  -3.1e-03],
           [  1.3e-01,   1.8e-01,   2.2e-02,  -1.2e-01,  -1.7e-03,   4.1e-03,   3.3e-02,  -3.7e-02,  -1.8e-01,  -2.5e-02],
           [ -1.5e-01,  -2.0e-01,  -2.4e-02,   1.4e-01,   1.9e-03,  -4.5e-03,  -3.7e-02,   4.1e-02,   2.0e-01,   2.8e-02],
           [ -7.3e-01,  -9.8e-01,  -1.2e-01,   6.8e-01,   9.6e-03,  -2.2e-02,  -1.8e-01,   2.0e-01,   1.0e+00,   1.4e-01],
           [ -1.0e-01,  -1.3e-01,  -1.6e-02,   9.3e-02,   1.3e-03,  -3.1e-03,  -2.5e-02,   2.8e-02,   1.4e-01,   1.9e-02]])




    covmat = xtx_rdd[1]/nsubvox
    plt.imshow(covmat)




    <matplotlib.image.AxesImage at 0x7f64387d5f90>




![png](noise_est_single_files/noise_est_single_22_1.png)


## Hadoop test


    def readPointBatch(iterator):
        import numpy as np
        strs = list(iterator)
        matrix = np.zeros((len(strs), D + 1))
        for i in xrange(len(strs)):
            matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype=np.float32, sep=' ')
        return [matrix]
    
    def readPoint(stt):
        ans = np.fromstring(stt.replace(',', ' '), dtype=np.float32, sep=' ')
        return (1, ans)



    data_text = 'data.txt'
    #local_path = '/root/all_b1000_1_data.csv'
    #hadoop_path = '/root/ephemeral-hdfs/bin/hadoop'
    #subprocess.check_output('bash ' + hadoop_path + ' fs -copyFromLocal ' + \
    #                        local_path + ' ' + data_text, shell=True)
    ## got error last time...
    partitions = 9
    brain_rdd = sc.textFile("data.txt", partitions).map(readPoint)


    def getS0S(tup):
        return tup[1][5:15]


    from operator import add
    nvox = brain_rdd.map(lambda x : 1).reduce(add)
    sum_S0S = brain_rdd.map(getS0S).reduce(add)


    nvox




    458566




    mu_S0S = sum_S0S/nvox


    def cc(v):
        v2 = v - mean(v)
        return np.outer(v2, v2).ravel()


    s0s_rdd = brain_rdd.map(getS0S).cache()
    xtx = s0s_rdd.map(cc).reduce(add)


    np.shape(xtx)




    (100,)




    cc_s0s = np.reshape(xtx, (10,10))/nvox
    plt.imshow(cc_s0s)




    <matplotlib.image.AxesImage at 0x7f0ca1500c10>




![png](noise_est_single_files/noise_est_single_33_1.png)



    np.set_printoptions(linewidth = 120, precision = 1)
    cc_s0s




    array([[  940.5,   367.8,   178.9,    49. ,   -43.9,  -229. ,  -294.6,  -389.1,  -284.2,  -295.3],
           [  367.8,   827.9,   281.6,    88.3,  -111.9,  -315.1,  -323.8,  -473.2,  -251.2,   -90.4],
           [  178.9,   281.6,   761.1,   163.2,  -100.7,  -341.3,  -277.6,  -483.3,  -171.3,   -10.6],
           [   49. ,    88.3,   163.2,   795.2,   104.7,  -170.1,  -214.1,  -335.9,  -259. ,  -221.2],
           [  -43.9,  -111.9,  -100.7,   104.7,   905.6,   160.9,   -49.1,   -75. ,  -407.7,  -382.8],
           [ -229. ,  -315.1,  -341.3,  -170.1,   160.9,  1043.7,    42.1,   469.8,  -301.1,  -359.9],
           [ -294.6,  -323.8,  -277.6,  -214.1,   -49.1,    42.1,   795.7,   188.5,   102.2,    30.6],
           [ -389.1,  -473.2,  -483.3,  -335.9,   -75. ,   469.8,   188.5,  1233.2,     6.1,  -141. ],
           [ -284.2,  -251.2,  -171.3,  -259. ,  -407.7,  -301.1,   102.2,     6.1,  1112.5,   453.7],
           [ -295.3,   -90.4,   -10.6,  -221.2,  -382.8,  -359.9,    30.6,  -141. ,   453.7,  1017. ]])




    def cc2(v):
        v2 = (v - mean(v))/mean(v)
        return np.outer(v2, v2).ravel()


    xtx = s0s_rdd.map(cc2).reduce(add)
    cc2_s0s = np.reshape(xtx, (10,10))/nvox


    plt.imshow(cc2_s0s)




    <matplotlib.image.AxesImage at 0x7f0ca13e5550>




![png](noise_est_single_files/noise_est_single_37_1.png)



    cc2_s0s




    array([[ 0.1, -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. ],
           [-0. ,  0.1, -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. ],
           [-0. , -0. ,  0.1, -0. , -0. , -0. , -0. , -0. , -0. , -0. ],
           [-0. , -0. , -0. ,  0.1, -0. , -0. , -0. , -0. , -0. , -0. ],
           [-0. , -0. , -0. , -0. ,  0.1, -0. , -0. , -0. , -0. , -0. ],
           [-0. , -0. , -0. , -0. , -0. ,  0.1, -0. , -0. , -0. , -0. ],
           [-0. , -0. , -0. , -0. , -0. , -0. ,  0.1, -0. , -0. , -0. ],
           [-0. , -0. , -0. , -0. , -0. , -0. , -0. ,  0.2, -0. , -0. ],
           [-0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. ,  0.1, -0. ],
           [-0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. ,  0.1]])




    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    import subprocess
    subprocess.check_output("git config --global user.name 'Charles EC2'", shell=True)
    subprocess.check_output("git commit -a -m 'py commit'", shell=True)




    '[master 130e9a7] py commit\n Committer: Charles EC2 <root@ip-172-31-20-178.us-west-2.compute.internal>\nYour name and email address were configured automatically based\non your username and hostname. Please check that they are accurate.\nYou can suppress this message by setting them explicitly:\n\n    git config --global user.name "Your Name"\n    git config --global user.email you@example.com\n\nAfter doing this, you may fix the identity used for this commit with:\n\n    git commit --amend --reset-author\n\n 2 files changed, 618 insertions(+), 68 deletions(-)\n'




    
