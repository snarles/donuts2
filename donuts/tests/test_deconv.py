import os
import numpy as np
import numpy.random as npr
import numpy.linalg as nla
import scipy.stats as spst
import scipy as sp
import scipy.optimize as spo
import numpy.testing as npt
import scipy.spatial.distance as spd
import donuts.deconv.utils as du
import donuts.deconv.ncx as ncx
import donuts.deconv.splines as spl

#def test_fail():
#    assert 1==2

def test_scalarize():
    assert type(du.scalarize(5)) == np.int64
    assert len(du.scalarize(np.ones(5))) ==5
    assert len(du.scalarize(np.ones((5,2)))) ==5

def test_column():
    assert np.shape(du.column(5))[1]==1
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

def test_simulate_signal_kappa():
    bvecs = du.geosphere(4)
    true_kappa = 1.5
    true_pos = bvecs[[1,10,11]]
    true_w = np.array([1.,1.,1.]).reshape((-1,1))

    # test for approximate gaussianity
    count = 0.0
    sigma = 1e-2
    flag = True
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,sigma)
    mu = 0.0 * np.array(y0)
    vr = 0.0 * np.array(y0)
    skw = 0.0 * np.array(y0)
    while flag:
        count = count + 1.0
        temp, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,sigma)
        mu = (count/(count+1))* mu + (1/(count+1))*(y1-y0)
        vr = (count/(count+1))* vr + (1/(count+1))*(y1-y0)**2
        skw = (count/(count+1))* skw + (1/(count+1))*(y1-y0)**3
        if (count >= 2) and (max(abs(mu)) < 1e-2) and (max(abs(vr - sigma**2)) < 1e-2) and (max(abs(skw)) < 1e-2):
            flag=False
        assert count < 100
        if count > 2:
            print("test deconv " + str(count) + " / 10 trials needed")
