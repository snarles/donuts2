
# coding: utf-8

# # Module

# In[1]:

import numpy as np


# In[4]:

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


# In[19]:

def csvrow2array(st):
    return np.array([float(s) for s in st.replace(',',' ').replace('  ',' ').split(' ')])


# In[41]:

def int2str(z):
    """
    Converts a int into an ASCII string, using a base-90 representation
    
    Inputs:
      z : int
    Outputs:
      str : the int represented as a string
    
    [Rule 1]: For z < 90, int2str(z) = chr(z + 33)
    
      int2str(0)  == chr(33) == '!'
      int2str(1)  == chr(34) == '"'
      int2str(2)  == chr(35) == '#'
      ...
      int2str(89) == chr(122) == 'z'
    
    [Rule 2]: Represent larger nonnegative numbers by joining digits with chr(123) == '{'
      90 == 1*90 + 0
      int2str(90) == chr(34) + chr(123) + chr(33) == '"{!'
                        '1'      'join'     '0'
    
      91 == 1*90 + 1 
      int2str(91) == chr(34) + chr(123) + chr(34) == '"{"'
                        '1'      'join'     '1'
    
      92 == 1*90 + 2 
      int2str(92) == chr(34) + chr(123) + chr(35) == '"#"'
                        '1'      'join'     '2'
    
    And so on for higher powers:
      16202 == 2*90**2 + 0*90 + 2
      int2str(16202) == chr(35) + chr(123) + chr(33) + chr(123) + chr(35) == '#{!{#'
                         '2'      'join'     '0'        'join'      '2'
                         
    [Rule 3]: Represent negative numbers by prepending chr(124) == '|'
      int2str(91) == '"{"'
      int2str(-91) == '|"{"'      
      
    [Note]:
      chr(123) == '{' is reserved for joining digits
      chr(124) == '|' is reserved for minus signs
      chr(125) == '}' is reserved for designating the -log_10 floating point precision (default 0)
      chr(126) == '~' is reserved for delimiting multiple CffStr
      E.g. '!}"{"' == int2str(0) + '}' + int2str(91) denotes 91.0/(10**0) == 91.0
           '#}"{"' == int2str(2) + '}' + int2str(91) denotes 91.0/(10**2) == 0.91
    """
    z = int(z)
    if (z < 0):
        return chr(124) + int2str(-z)
    if (z < 90):
        return chr(z+33)
    else:
        resid = int(z % 90)
        z = int(z-resid)/90
        return int2str(z)+chr(90+33)+chr(resid+33)
    
def ints2str(zs):
    """
    Converts an array or list of ints to a string.
    
    Inputs:
      zs : list of ints
    Outputs:
      str : the list of ints represented as a string
    
    Example:
      int2str(100) == '"{+'
      int2str(200) == '#{5'
      ints2str([100, 200]) == '"{+#{5'
    See int2str(z)
    """
    return ''.join(int2str(z) for z in zs)

def floats2str(xs, intRes):
    """
    Converts an array or list of floats to a string.
    
    Inputs:
      xs : list of floats
      intRes: controls the precision
              the numerical precision will be equal to 1/(2**intRes)
    Outputs:
      str : the list of floats represented as a string
    """
    zs = np.array(xs * float(10**intRes), dtype=int)
    return int2str(intRes) + '}' + ints2str(zs)

def str2floats(stt):
    """
    Recovers the list of ints from the string representation.
    See ints2str(zs)
    
    Inputs:
      st: String composed of characters in [ord(33), ... ,ord(124)] == ['!',...,'~']
    """
    st = stt.split('}')
    if len(st)==1:
        return np.array(str2ints(st[-1]), dtype=float)
    if len(st)==2:
        intRes = str2ints(st[0])[0]
        return np.array(str2ints(st[-1]), dtype=float)/float(10**intRes)

def str2ints(st, maxlen = -1):
    """
    Recovers the list of ints from the string representation.
    See ints2str(zs)
    
    Inputs:
      st: String composed of characters in [ord(33), ... ,ord(124)] == ['!',...,'|']
    """
    os = [ord(c[0])-33 for c in st]
    assert(min(os) >= 0 and max(os) <= 91) 
    zs = []
    counter = 0
    if maxlen == -1:
        maxlen = len(os)
    while (counter < len(os)) and (len(zs) < maxlen):
        if os[counter] == 90: # 'join' symbol
            zs[-1] = zs[-1] * 90 + sign(zs[-1]) * os[counter + 1]
            counter = counter + 1
        elif os[counter] == 91: # 'minus' symbol
            zs.append(-os[counter+1])
            counter = counter + 1
        else:
            zs.append(os[counter])
        counter = counter + 1
    return zs


# In[42]:

class CffStr(str):
    """
    A class representing a float array represented as a compressed str.
    See functions ints2str, str2ints
    Intialization:
      Option 1) Initialize using a string
      Option 2) Initialize using an integer or int array
      Option 3) Initialize using a dictionary (initOpts)
                Dictionary values:
                  value (optional): the compressed data string (of type str)
                                    if NOT included, must include 'coords' and 'floats' OR 'coords' and 'ints'
                  ints (optional):   an integer list or array
                  floats (optional): floating-point values
                                     if included, must also include intRes
                  intRes (optional): the numerical precision is parts per 10**intRes
    """
    
    def __new__(cls, initOpts):
        value = ''
        if type(initOpts) == str:
            value = initOpts
        if type(initOpts) == int:
            value = int2str(initOpts)
        if type(initOpts) in [list, tuple, numpy.ndarray]:
            value = ints2str(initOpts)
        if type(initOpts) == dict:
            if initOpts.has_key('value'):
                value = initOpts['value']
            if initOpts.has_key('ints'):
                value = ints2str(initOpts['ints'])
            if initOpts.has_key('floats'):
                assert(initOpts.has_key('intRes'))
                value = floats2str(initOpts['floats'], initOpts['intRes'])
        return str.__new__(cls, value)
    
    def getValue(self):
        return str(self)
    
    def getInts(self):
        st = str(self).split('}')[-1]
        return str2ints(st)
    
    def getFloats(self):
        return str2floats(str(self))
    
class MultiCffStr(CffStr):
    """
    A class representing multiple cff strings
    See functions ints2str, str2ints
    Intialization:
      Option 1) Initialize using a string
      Option 2) Initialize using a list of cffs
    """
    def __new__(cls, initOpts):
        value = ''
        if type(initOpts) == str:
            value = initOpts
        if type(initOpts) == list:
            value = '~'.join(initOpts)
        return str.__new__(cls, value)
    
    def getCffs(self):
        return [CffStr(st) for st in str(self).split('~')]
    
    def getValue(self):
        return str(self)
    
    def getInts(self):
        return self.getCffs()[0].getInts()
    
    def getFloats(self):
        return self.getCffs()[0].getFloats()


# In[43]:

if __name__ == "__main__":
    c1 = CffStr({'floats': np.array([-5.1,2.2]), 'intRes': 2})
    c2 = CffStr({'ints': [1, 2, 3]})
    c3 = CffStr((-121, 34122))
    
    m1 = MultiCffStr([c1, c2, c3])
    
    print(c1.getFloats())
    print(c2.getFloats())
    print(c3.getInts())
    
    print(m1.getCffs())


# In[60]:

class Voxel(tuple):
    """
    A class representing a single 3-dimensional voxel.
    Attributes are coordinates (key) and data. Optionally: cached data.
    Internally, it uses characters to compress integer-valued data.
    Initialize with a compressed string of CFF format.
    Intialization:
      Option 1) Initialize using a compressed data string (of type str)
      Option 2) Initialize using a (Key, Value) str tuple (as is done by Spark automatically)
                Same as option 1, with Value used as string
      Option 3) Initialize using a dictionary (initOpts)
                Dictionary values:
                  intRes: the numerical precision for floats
                INCLUDE:
                  csv_row: a row read from a CSV file
                  ncoords: the number of coordinates in the CSV file, default 3
                OR INCLUDE:
                  coords: a tuple containing the coords
                  floats: floating-point values
    """
    def __new__(cls, initOpts):
        if type(initOpts) == str:
            temp = initOpts.split('~')
            key = CffStr(temp[0])
            if len(temp) > 2:
                value = MultiCffStr('~'.join(temp[1:]))
            else:
                value = CffStr(temp[1])
        if type(initOpts) == tuple: # constructed via Spark deserialization
            key = CffStr(initOpts[0])
            value = CffStr(initOpts[1])
        if type(initOpts) == dict:
            if initOpts.has_key('csv_row'):
                ncoords = initOpts.get('ncoords', 3)
                temp = csvrow2array(initOpts['csv_row'])
                initOpts['coords'] = temp[:ncoords]
                initOpts['floats'] = temp[ncoords:]
            key = CffStr(initOpts['coords'])
            value = CffStr(initOpts)
        kv = tuple((key, value))
        assert(len(kv)==2)
        return super(Voxel, cls).__new__(cls, tuple(kv))
    
    # creates a new voxel from float data
    def convertData(self, floats, intRes, coords = None):
        if coords is None:
            coords = self.getCoords()
        newVoxel = Voxel({'coords': self.getCoords(), 'floats': floats, 'intRes': intRes})
        return newVoxel
    
    def getCoords(self):
        coords = tuple(self[0].getInts())
        return coords
    
    def getData(self):
        return self[1].getFloats()
    
    def bareCopy(self):
        return (str(self[0]), str(self[1]))
    
    def toString(self): # used for writing to flat files
        return str(self[0])+'~'+str(self[1])
    
    def toCsvString(self, delimiter=','): # used for writing to flat files
        return delimiter.join([str(v) for v in np.hstack([self.getCoords(), self.getData()])])


# In[61]:

if __name__ == "__main__":
    m = Voxel({'coords': (1, 2, 3), 'floats': np.array([1.12, 3.3, -4.5]), 'intRes': 2})
    print(m.getCoords())
    print(m.getData())
    bc = m.bareCopy()
    print(bc)
    m_clone = Voxel(bc)
    print(m, m_clone)
    print(m.toCsvString())


# In[62]:

if __name__ == "__main__":
    import numpy.random as npr
    from StringIO import StringIO
    si = StringIO()
    # define functions used in testing
    nvox = 100
    def gen_vox():
        coords = npr.randint(0, 10, 3)
        data = npr.randn(20)
        return np.hstack([coords, data])
    # simulate text file
    rawvoxes= np.array([gen_vox() for ii in range(nvox)])
    np.savetxt(si, rawvoxes)
    rawdata = si.getvalue().strip().split('\n')
    voxes = [Voxel({'intRes': 3, 'csv_row': rawdat}) for rawdat in rawdata]


# # Testing in Spark

# In[4]:

if __name__ == "__main__":
    import numpy.random as npr
    from donuts.spark.classes import Voxel
    from StringIO import StringIO
    si = StringIO()
    # define functions used in testing
    nvox = 100
    def gen_vox():
        coords = npr.randint(0, 10, 3)
        data = npr.randn(20)
        return np.hstack([coords, data])
    # simulate text file
    rawvoxes= np.array([gen_vox() for ii in range(nvox)])
    np.savetxt(si, rawvoxes)
    rawdata = si.getvalue().strip().split('\n')
    voxes = [Voxel({'intRes': 3, 'csv_row': rawdat}) for rawdat in rawdata]


# In[ ]:



