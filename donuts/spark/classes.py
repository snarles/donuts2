
# coding: utf-8

# # Module

# In[ ]:

import numpy as np
import time
import numpy.random as npr
from StringIO import StringIO
import math
import numpy.testing as npt
import numpy.random as npr
from donuts.spark.fake import *


# ### Numpy conversion functions

# In[ ]:

def np_to_str(a):
    """
    Converts an np array to a binary string without newline
    """
    a = np.array(a, dtype = np.float16)
    si = StringIO()
    np.save(si, a)
    st =  si.getvalue()
    return st

def str_to_np(st): 
    """
    Converts a binary string to an np array
    """
    return np.load(StringIO(st))

def part_to_str(iterator):
    arr = np.vstack(list(iterator))
    st = np_to_str(arr)
    return [st]

def str_to_part(iterator):
    sts = list(iterator)
    return np.vstack([str_to_np(st) for st in sts])


# ### Helper functions for data classes

# In[203]:

def partition_array3(arr, sz):
    """
    Partitions a 4d array into a list of 4d arrays by the first 3 dimensions
    """
    dims = np.shape(arr)
    dim_parts = tuple([int(math.ceil(float(dims[i])/sz[i])) for i in range(3)])
    nparts = np.prod(dim_parts)
    temp = np.unravel_index(range(nparts), dim_parts)
    lcorners = np.array([sz[i] * temp[i] for i in range(3)]).T
    ucorners = np.array([sz[i] * temp[i] + sz[i] for i in range(3)]).T
    for i in range(3):
        ucorners[ucorners[:, i] > dims[i], i] = dims[i]
    subarrs = [(tuple(lc), arr[lc[0]:uc[0], lc[1]:uc[1], lc[2]:uc[2]])                for (lc, uc) in zip(lcorners, ucorners)]
    return subarrs

def _unpickled_to_subarr(tup):
    return (tup[0], str_to_np(tup[1]))

def _subarr_to_compressed(tup):
    return (tup[0], np_to_str(tup[1]))

def _apply_func(tup, f, inds):
    if inds is None:
        return (tup[0], np.apply_over_axes(f, tup[1], [3]))
    else:
        return (tup[0], np.apply_over_axes(f, tup[1][:, :, :, inds], [3]))
    
def _apply_funcs(tup, fs, inds):
    if inds is None:
        arrs = [np.apply_over_axes(f, tup[1], [3]) for f in fs]
        arr = np.concatenate(arrs, 3)
    else:
        arrs = [np.apply_over_axes(f, tup[1][:, :, :, inds], [3]) for f in fs]
        arr = np.concatenate(arrs, 3)
    return (tup[0], arr)

def _u_filter_c(tup, inds):
    return (tup[0], np_to_str(str_to_np(tup[1])[:, :, :, inds]))

def _aug_key(tup, ind):
    return (tup[0], [ind, tup[1]])

def _sort_combine(tup):
    ll = len(tup[1])/2
    itups = [tup[1][2*i] for i in range(ll)]
    os = sorted(range(ll), key=itups.__getitem__)
    sorted_els = [tup[1][2*i+1] for i in os]
    arrs = [str_to_np(el) for el in sorted_els]
    arr = np.concatenate(arrs, 3)
    return (tup[0], arr)


# ### Specialized RDD container classes

# In[ ]:




# In[204]:

class VoxelPartition:
    
    def __init__(self, rdd = None, cont = None, picklef = None, arr = None, sz = None,                  parts = 10, picklefs = None, inds = None):
        # 1. Set up needded parameters
        if cont is None:
            cont = FakeSparkContext()
        self.cont = cont
        if sz is None:
            sz = (10, 10, 10)
        self.sz = sz
        if parts is None:
            parts = 10
        self.parts = parts
        # 2. form RDD
        if rdd is not None:
            self.rdd = rdd
        elif arr is not None:
            self.rdd = cont.parallelize(partition_array3(arr, sz), parts)
        elif picklefs is not None:
            rdds = []
            for ind in range(len(picklefs)):
                if inds is None:
                    rdd = cont.pickleFile(picklefs[ind], parts).map(lambda x : _aug_key(x, ind))
                else:
                    rdd = cont.pickleFile(picklefs[ind], parts).map(lambda x : _u_filter_c(x, inds)).                        map(lambda x : _aug_key(x, ind))
                rdd.takeSample(False, 1) # Force evaluation
                rdds[ind] = rdd
            new_rdd = cont.union(rdds).                combineByKey(lambda x : x, lambda x, y : x + y, lambda x, y: x + y)
            self.rdd = new_rdd.map(_sort_combine)                
        elif picklef is not None:
            self.rdd = cont.pickleFile(picklef, parts).map(_unpickled_to_subarr)
    
    # computes a function on each voxel and returns an RDD with that result
    def compute_quantity(self, func, inds = None):
        new_rdd = self.rdd.map(lambda x : _apply_func(x, func, inds))
        return VoxelPartition(cont = self.cont, sz = self.sz, parts = self.parts, rdd = new_rdd)
    
    def compute_quantities(self, lfuncs, inds = None):
        new_rdd = self.rdd.map(lambda x : _apply_funcs(x, lfuncs, inds))
        return VoxelPartition(cont = self.cont, sz = self.sz, parts = self.parts, rdd = new_rdd)
    
    def save_as_pickle_file(self, fname):
        self.rdd.map(_subarr_to_compressed).saveAsPickleFile(fname)
    
    
    


# In[ ]:




# In[ ]:




# In[ ]:




# # Testing

# In[2]:

if __name__ == "__main__":
    import numpy as np
    import numpy.random as npr
    from donuts.spark.classes import *
    dims = (25, 30, 20, 100)
    arr1 = npr.normal(0, 1, dims)
    arr2 = npr.normal(10, 1, dims)
    sz = (10, 10, 10)
    VoxelPartition(arr=arr1, sz=sz, cont = sc).save_as_pickle_file('arr1.pickle')
    VoxelPartition(arr=arr2, sz=sz, cont = sc).save_as_pickle_file('arr2.pickle')
    vp = VoxelPartition(cont = sc, picklefs = ['arr1.pickle', 'arr2.pickle'], inds = range(1,3))
#    means = voxPart.compute_quantities([np.mean, np.var], range(1,3))
#    means = voxPart.compute_quantity(np.mean, range(1,3))
#    means.save_as_pickle_file('means.pickle')
#    means2 = VoxelPartition(cont = sc, picklef = 'means.pickle')
    print(np.shape(vp.rdd.first()[1]))


# In[ ]:



