
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

class TestClass(object):
    a = 5
    def __init__(self, a):
        self.a = a
    def getvalue():
        return a
    
class Foo(tuple):

    def __new__ (cls, a):
        return super(Foo, cls).__new__(cls, tuple(a[0][0:2], a[1][0:2]))

    def __init__(self, a):
        super(Foo, self).__init__(a)
        self.a=a


# In[ ]:

class Voxel(tuple):
    'A class representing a single 3-dimensional voxel.      Attributes are coordinates (key) and data. Optionally: cached data.    Internally, it uses characters to compress integer-valued data.     Initialize with a compressed string of CFF format.'
    intRes = 10000
    offset = 0
    ncoords = 3
    
    def __new__(cls, ncoords, initString):
        ints = str2ints(initString)
        coords = tuple(ints[0:ncoords]) # edit this when changing the number of coords
        key = ints2str(coords)
        kv = (key, initString)
        return super(Voxel, cls).__new__(cls, tuple(kv))
    
    def __init__(self, ncoords, initString):
        super(Voxel, self).__init__(ncoords, initString)
        self.ncoords = ncoords
        
    def convertInts(self, ints):
        newDataString = key + ints2str(ints)
        return Voxel(newDataString)
    
    def convertFloats(self, floats):
        ints = np.hstack([floats[range(ncoords)], np.array(floats[3:] * self.intRes)])
        return self.convertInts(ints)
        
    def getCoords(self):
        ints = str2ints(self.key)
        coords = tuple(ints[range(ncoords)])
        return coords

    def getIntData(self):
        return np.array(str2ints(self[1]), dtype=int)
    
    def getFloatData(self):
        ints = np.array(str2ints(self[1]), dtype=int)
        return np.hstack([ints[0:3], np.array(ints[3:]/float(self.intRes))])-offset
    
    def setConversion(self, intRes=None, offset=None, ncoords=None):
        if intRes is not None:
            self.intRes = intRes
        if offset is not None:
            self.offset = offset
        if ncoords is not None:
            self.ncoords = ncoords

