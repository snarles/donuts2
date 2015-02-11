
# coding: utf-8

# # Module

# In[1]:

import numpy as np
import time
import numpy.random as npr
from StringIO import StringIO
import math
import numpy.testing as npt
import numpy.random as npr
from donuts.spark.fake import *


# ### Numpy conversion functions

# In[2]:

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

# In[3]:

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

def np_to_txt(tup):
    """
    Converts an np array to unraveled txt string
    """
    dims = np.shape(tup[1])
    a = np.array(tup[1], dtype = np.float16).ravel()
    si = StringIO()
    np.savetxt(si, np.hstack([tup[0], dims, a]), fmt = '%.6e')
    st =  si.getvalue()
    st = st.replace('\n',' ')
    return st

def txt_to_np(st): 
    """
    Converts a binary string to an np array
    """
    st = st.replace(' ','\n')
    seq = np.loadtxt(StringIO(st))
    dims = seq[3:7]
    coords = seq[:3]
    data = seq[7:]
    ndata = np.prod(dims)
    tup = (tuple(np.array(coords, dtype=int)), np.reshape(data[:ndata], tuple(dims)))
    return tup


# In[22]:

if __name__ == "__main__":
    import numpy as np
    import numpy.random as npr
    #from donuts.spark.classes import *
    import os
    import time
    # function arguments
    dims = (40, 40, 40, 20)
    arr = npr.normal(0, 1, dims)
    sz = (10, 10, 10)
    tempf = 'temp1.txt'
    # function 1
    os.system('rm temp1.txt')
    t1 = time.clock()
    arrs = partition_array3(arr, sz)
    strs = [np_to_txt(tup) for tup in arrs]
    f = open(tempf, 'w')
    f.write('\n'.join(strs))
    f.close()
    time1 = time.clock() - t1
    # function 2
    tempf = 'temp2.txt'
    os.system('rm temp2.txt')
    t1 = time.clock()
    arrs = partition_array3(arr, sz)
    lenmax = np.prod(sz) * np.shape(arr)[3]
    flatarr = np.array([np.hstack([tup[0], np.shape(tup[1]), tup[1].ravel(),        np.zeros(lenmax - np.prod(np.shape(tup[1])))]) for tup in arrs])
    np.savetxt(tempf, flatarr)
    time2 = time.clock() - t1
    print((time1, time2))


# In[26]:

if __name__ == "__main__":
    # function 1
    f = open('temp1.txt', 'r')
    st1 = f.read().split('\n')
    f.close()
    f = open('temp2.txt', 'r')
    st2 = f.read().split('\n')
    f.close()
    print(str(txt_to_np(st1[0]))[0:100])
    print(str(txt_to_np(st2[0]))[0:100])
    print(str(arrs[0])[0:100])


# ### Specialized RDD container classes

# In[4]:

class VoxelPartition:
    
    def __init__(self, rdd = None, cont = None, picklef = None, sz = None,                  parts = 10, picklefs = None, inds = None, textf = None):
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
        elif picklefs is not None:
            rdds = []
            for ind in range(len(picklefs)):
                if inds is None:
                    evalst = 'cont.pickleFile(picklefs[indx], parts).map(lambda x : _aug_key(x, indx))'
                else:
                    evalst = 'cont.pickleFile(picklefs[indx], parts).map(lambda x : _u_filter_c(x, inds)).                        map(lambda x : _aug_key(x, indx))'
                rdd = eval(evalst.replace('indx', str(ind)))
                rdds.append(rdd)
            new_rdd = cont.union(rdds).                combineByKey(lambda x : x, lambda x, y : x + y, lambda x, y: x + y)
            self.rdd = new_rdd.map(_sort_combine)                
        elif picklef is not None:
            self.rdd = cont.pickleFile(picklef, parts).map(_unpickled_to_subarr)
        elif textf is not None:
            rt = cont.textFile(textf, parts)
            self.rdd = rt.map(txt_to_np)

    # computes a function on each voxel and returns an RDD with that result
    def compute_quantity(self, func, inds = None):
        new_rdd = self.rdd.map(lambda x : _apply_func(x, func, inds))
        return VoxelPartition(cont = self.cont, sz = self.sz, parts = self.parts, rdd = new_rdd)
    
    def compute_quantities(self, lfuncs, inds = None):
        new_rdd = self.rdd.map(lambda x : _apply_funcs(x, lfuncs, inds))
        return VoxelPartition(cont = self.cont, sz = self.sz, parts = self.parts, rdd = new_rdd)
    
    def save_as_pickle_file(self, fname):
        self.rdd.map(_subarr_to_compressed).saveAsPickleFile(fname)
    
    
    


# ### Script for storing an array to txt

# In[5]:

import os
def convscript(arr, tempf = 'temp.txt', sz = (10, 10, 10), hadoop_dir = '/root/ephemeral-hdfs'):
    """
    hadoop_dir: no slash on the end, don't include /bin
    cont: spark context
    """
    arrs = partition_array3(arr, sz)
    arrs = partition_array3(arr, sz)
    lenmax = np.prod(sz) * np.shape(arr)[3]
    flatarr = np.array([np.hstack([tup[0], np.shape(tup[1]), tup[1].ravel(),        np.zeros(lenmax - np.prod(np.shape(tup[1])))]) for tup in arrs])
    os.chdir(hadoop_dir + '/bin')
    np.savetxt(tempf, flatarr)
    print('Wrote to file...')
    os.system('./hadoop fs -mkdir '+tempf+' temp/'+tempf)
    os.system('./hadoop fs -rmr temp/'+tempf)
    os.system('./hadoop fs -put '+tempf+' temp/'+tempf)
    print('Copied to hadoop... temp/' + tempf)
    os.system('rm '+tempf)
    print('Cleaning up...')
    return


# In[27]:

if __name__ == "__main__":
    import numpy as np
    import numpy.random as npr
    #from donuts.spark.classes import *
    import os
    import time
    # function arguments
    dims = (40, 40, 40, 20)
    arr = npr.normal(0, 1, dims)
    sz = (10, 10, 10)
    convscript(arr, 'temp.txt')


# In[28]:

if __name__ == "__main__":
    rdd = sc.textFile('temp/temp.txt', 20).map(txt_to_np)
    tup = rdd.takeSample(False, 1)[0]
    coords = tup[0]
    print(arr[coords[0], coords[1], coords[2], :])
    print(tup[1][0, 0, 0, :])


# ```
# rdds = []
# parts = 20
# picklefs = ['arr1.pickle', 'arr2.pickle']
# cont = sc
# ind = -1
# ind = ind + 1
# evalst = 'cont.pickleFile(picklefs[indx], parts).map(lambda x : _aug_key(x, indx))'
# rdd = eval(evalst.replace('indx', str(ind)))
# smp = rdd.takeSample(False, 1)
# rdds.append(rdd)
# new_rdd = cont.union(rdds).combineByKey(lambda x : x, lambda x, y : x + y, lambda x, y: x + y)
# smp = cont.union(rdds).takeSample(False, 10)
# smp[0][0]
# ```

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


# In[3]:

if __name__ == "__main__":
    import numpy as np
    import numpy.random as npr
    from donuts.spark.classes import *
    import os
    dims = (40, 40, 40, 20)
    arr1 = npr.normal(0, 1, dims)
    sz = (10, 10, 10)
    convscript(arr1, 'temp1.txt')
    vp = VoxelPartition(textf = 'temp/temp1.txt', cont=sc)
    tup = vp.rdd.takeSample(False, 1)[0]
    coords = tup[0]
    print(arr1[coords[0], coords[1], coords[2], 0:100:10])
    print(tup[1][0, 0, 0, 0:100:10])
    os.chdir('/root/ephemeral-hdfs/bin')
    os.system('./hadoop fs -rmr arr1.pickle')
    vp.save_as_pickle_file('arr1.pickle')
    arr2 = npr.normal(0, 1, dims)
    convscript(arr2, 'temp2.txt')
    vp = VoxelPartition(textf = 'temp/temp2.txt', cont=sc)
    tup = vp.rdd.takeSample(False, 1)[0]
    coords = tup[0]
    print(arr2[coords[0], coords[1], coords[2], 0:100:10])
    print(tup[1][0, 0, 0, 0:100:10])
    os.chdir('/root/ephemeral-hdfs/bin')
    os.system('./hadoop fs -rmr arr2.pickle')
    vp.save_as_pickle_file('arr2.pickle')


# In[4]:

if __name__ == "__main__":
    main_arr = np.concatenate([arr1, arr2])
    vpm = VoxelPartition(picklefs = ['arr1.pickle', 'arr2.pickle'], cont = sc)


# In[ ]:



