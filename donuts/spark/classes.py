
# coding: utf-8

# In[1]:

import numpy as np


# In[1]:

class TestClass(object):
    a = 5
    def __init__(self, a):
        self.a = a
    def getvalue():
        return a
    
class TestTuple(tuple):
    def __new__ (cls, a):
        return super(TestTuple, cls).__new__(cls, tuple([a[:2], a[:2]]))

    def __init__(self, a):
        super(TestTuple, self).__init__(a)
        self.a=a
    
    def getvalue(self):
        return a + a


# In[5]:

def int2str(z):
    if (z < 90):
        return chr(z+33)
    else:
        resid = int(z % 90)
        z = int(z-resid)/90
        return int2str(z)+chr(90+33)+chr(resid+33)
    
def ints2str(zs):
    return ''.join(int2str(z) for z in zs)

def str2ints(st):
    os = [ord(c)-33 for c in st]
    zs = []
    counter = 0
    while counter < len(os):
        if os[counter] == 90:
            zs[-1] = zs[-1] * 90 + os[counter + 1]
            counter = counter + 1
        else:
            zs.append(os[counter])
        counter = counter + 1
    return zs

def csvrow2array(st):
    return np.array([float(s) for s in st.replace(',',' ').replace('  ',' ').split(' ')])

def str2array(st):
    pts = st.split('|')
    arr = np.array([str2ints(pt) for pt in pts]).T
    return arr


# In[60]:

class Voxel(tuple):
    'A class representing a single 3-dimensional voxel.      Attributes are coordinates (key) and data. Optionally: cached data.    Internally, it uses characters to compress integer-valued data.     Initialize with a compressed string of CFF format.'
    intRes = 10000.0
    minVal = -100.0
    
    def __new__(cls, initString):
        nc = 3 # edit this when changing the number of coords
        ints = str2ints(initString)
        coords = tuple(ints[:nc])
        key = ints2str(coords)
        kv = (key, initString)
        return super(Voxel, cls).__new__(cls, tuple(kv))
            
    # creates a new voxel with the same coordinates, but different data
    def convertData(self, floats, intRes=10000.0, minVal=-100.0):
        ints = np.array((floats - minVal) * intRes, dtype=int)
        ints[ints < 0] = 0
        newDataString = self[0] + ints2str(ints)
        newVoxel = Voxel(newDataString)
        newVoxel.setConversion(intRes, minVal)
        return newVoxel
    
    def getCoords(self):
        nc = 3 # edit this when changing the number of coords
        ints = str2ints(self[0])
        coords = tuple(ints[:nc])
        return coords
    
    def getData(self):
        nc = 3 # edit this when changing the number of coords
        ints = np.array(str2ints(self[1]), dtype=int)
        return np.array(ints[nc:]/float(self.intRes))+self.minVal
    
    def setConversion(self, intRes=None, minVal=None):
        if intRes is not None:
            self.intRes = intRes
        if minVal is not None:
            self.minVal = minVal


# In[ ]:



