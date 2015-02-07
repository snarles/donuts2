
# Analysis of CNI data (1)


    import cvxopt as cvx
    import numpy as np
    import numpy.random as npr
    import scipy as sp
    import scipy.optimize as spo
    import matplotlib.pyplot as plt
    import subprocess


    sc




    <pyspark.context.SparkContext at 0x7f9c4ba11550>




    bvecs = np.loadtxt("/root/chris1_bvec", delimiter = " ")
    bvecs = reshape(bvecs, (3, -1)).T


    n_d = np.shape(bvecs)[0]
    n_d




    150




    inds_b0 = [i for i in range(2, n_d) if sum(bvecs[i, :]**2)==0]
    inds_b0




    [16, 31, 46, 61, 75, 90, 105, 120, 135, 149]




    partitions = 100
    raw = sc.textFile("part00,part01,part02,part03,part04,part05", partitions)


    def coords2key(coords):
        return coords[2] + coords[1]*200 + coords[0]*200*200
    
    def key2coords(key):
        c3 = key % 200
        c2 = ((key-c3)/200) % 200
        c1 = (key - c3 - 200*c2)/(200*200)
        return np.array([c1, c2, c3])
    
    def readVoxStr(stt):
        vec = np.fromstring(stt.replace(',', ' '), dtype=np.float32, sep=' ')
        coords = np.array(vec[0:3], dtype=int)
        key=coords2key(coords)
        return (key, vec[3:])


    pts = raw.map(readVoxStr).cache()


    coords = pts.map(lambda x: x[0]).map(key2coords)


    def max_red(x, y):
        return np.array([max(x[0], y[0]), max(x[1], y[1]), max(x[2], y[2])])
    
    def min_red(x, y):
        return np.array([min(x[0], y[0]), min(x[1], y[1]), min(x[2], y[2])])


    maxcoord = coords.reduce(max_red)
    mincoord = coords.reduce(min_red)
    maxcoord, mincoord




    (array([27, 70, 23]), array([0, 0, 0]))




    def lenrange(tup):
        return np.array([len(tup[1]), len(tup[1])], dtype=int)
    
    def range_red(rng1, rng2):
        return np.array([min(rng1[0], rng2[0]), max(rng1[1], rng1[1])], dtype=int)


    length_range = pts.map(lenrange).reduce(range_red)


    length_range




    array([4800, 4800])




    from operator import add
    pts.map(lambda x: x[0]).reduce(add)




    3201668




    nz = pts.filter(lambda x: x[0] != 0)


    st0 = raw.takeSample(False, 1)


    st0 = st0[0]
    nz = raw.filter(lambda x: x != st0)


    nzs = nz.takeSample(False, 10)


    len(nzs)




    3




    
