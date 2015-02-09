
# coding: utf-8

# # Module

# In[28]:

import numpy as np
import math


# In[9]:

def csv_row_to_array(st):
    return np.array([float(s) for s in st.replace(',',' ').replace('  ',' ').split(' ')])

def int_to_str(z):
    """
    Converts a int into an ASCII string, using a base-90 representation
    
    Inputs:
      z : int
    Outputs:
      str : the int represented as a string
    
    [Rule 1]: For z < 90, int2str(z) = chr(z + 33)
    
      int_to_str(0)  == chr(33) == '!'
      int_to_str(1)  == chr(34) == '"'
      int_to_str(2)  == chr(35) == '#'
      ...
      int_to_str(89) == chr(122) == 'z'
    
    [Rule 2]: Represent larger nonnegative numbers by joining digits with chr(123) == '{'
      90 == 1*90 + 0
      int_to_str(90) == chr(34) + chr(123) + chr(33) == '"{!'
                        '1'      'join'     '0'
    
      91 == 1*90 + 1 
      int_to_str(91) == chr(34) + chr(123) + chr(34) == '"{"'
                        '1'      'join'     '1'
    
      92 == 1*90 + 2 
      int_to_str(92) == chr(34) + chr(123) + chr(35) == '"#"'
                        '1'      'join'     '2'
    
    And so on for higher powers:
      16202 == 2*90**2 + 0*90 + 2
      int_to_str(16202) == chr(35) + chr(123) + chr(33) + chr(123) + chr(35) == '#{!{#'
                         '2'      'join'     '0'        'join'      '2'
                         
    [Rule 3]: Represent negative numbers by prepending chr(124) == '|'
      int_to_str(91) == '"{"'
      int_to_str(-91) == '|"{"'      
      
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
        return chr(124) + int_to_str(-z)
    if (z < 90):
        return chr(z+33)
    else:
        resid = int(z % 90)
        z = int(z-resid)/90
        return int_to_str(z)+chr(90+33)+chr(resid+33)
    
def ints_to_str(zs):
    """
    Converts an array or list of ints to a string.
    
    Inputs:
      zs : list of ints
    Outputs:
      str : the list of ints represented as a string
    
    Example:
      int_to_str(100) == '"{+'
      int_to_str(200) == '#{5'
      ints_to_str([100, 200]) == '"{+#{5'
    See int2str(z)
    """
    return ''.join(int_to_str(z) for z in zs)

def floats_to_str(xs, precision):
    """
    Converts an array or list of floats to a string.
    
    Inputs:
      xs : list of floats
      precision: the numerical precision will be equal to 1/(10**intRes)
    Outputs:
      str : the list of floats represented as a string
    """
    zs = np.array(xs * float(10**precision), dtype=int)
    return int_to_str(precision) + '}' + ints_to_str(zs)

def multi_floats_to_str(xs_s, precision_s):
    assert(len(xs_s)== len(precision_s))
    return '~'.join([floats_to_str(xs_s[ii], precision_s[ii]) for ii in range(len(xs_s))])

def str_to_floats(stt):
    """
    Recovers the list of ints from the string representation.
    See ints_to_str(zs)
    
    Inputs:
      st: String composed of characters in [ord(33), ... ,ord(124)] == ['!',...,'~']
    """
    st = stt.split('}')
    if len(st)==1:
        return np.array(str_to_ints(st[-1]), dtype=float)
    if len(st)==2:
        precision = str_to_ints(st[0])[0]
        return np.array(str_to_ints(st[-1]), dtype=float)/float(10**precision)
    
def str_to_multi_floats(sts):
    """
    Recovers a list of float arrays
    """
    return [str_to_floats(st) for st in sts.split('~')]
    
def str_to_ints(st, maxlen = -1):
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
            zs[-1] = zs[-1] * 90 + math.copysign(1, zs[-1]) * os[counter + 1]
            counter = counter + 1
        elif os[counter] == 91: # 'minus' symbol
            zs.append(-os[counter+1])    
            counter = counter + 1
        else:
            zs.append(os[counter])
        counter = counter + 1
    return [int(z) for z in zs]


# In[22]:

class CffStr(str):
    """
    A class representing a float array (or list thereof) represented as a compressed str.
    See functions ints2str, str2ints
    Intialization:
      Option 1) Initialize using a string
      Option 2) Initialize using an integer or int array
      Option 3) Initialize using a dictionary (initOpts)
                Dictionary values:
                  value (optional): the compressed data string (of type str)
                                    if NOT included, must include 'coords' and 'floats' OR 'coords' and 'ints'
                  cffs (optional): a list of cffs or strings
                  ints (optional):   an integer list or array
                  floats (optional): floating-point values
                                     if included, must also include intRes
                  multi_floats (optional): an array or list of floats
                  precision (optional): the numerical precision is parts per 10**intRes
                  
    """
    
    def __new__(cls, initOpts):
        value = 'emptyCFF'
        if type(initOpts) == str:
            value = initOpts
        elif type(initOpts) == unicode:
            value = str(initOpts)
        elif type(initOpts) == int:
            value = int_to_str(initOpts)
        elif str(type(initOpts)) in ["<type 'numpy.ndarray'>", "<type 'list'>", "<type 'tuple'>"]:
            value = ints_to_str(initOpts)
        elif type(initOpts) == dict:
            if initOpts.has_key('value'):
                value = initOpts['value']
            elif initOpts.has_key('cffs'):
                value = '~'.join(initOpts['cffs'])
            elif initOpts.has_key('ints'):
                value = ints_to_str(initOpts['ints'])
            elif initOpts.has_key('floats'):
                assert(initOpts.has_key('precision'))
                value = floats_to_str(initOpts['floats'], initOpts['precision'])
            elif initOpts.has_key('multi_floats'):
                assert(initOpts.has_key('precision'))
                value = multi_floats_to_str(initOpts['multi_floats'], initOpts['precision'])
        elif 'CffStr' in str(type(initOpts)):
            value = str(initOpts)
        return str.__new__(cls, value)
    
    def get_value(self):
        return str(self)
    
    def get_ints(self):
        assert('~' not in str(self))
        st = str(self).split('}')[-1]
        return str_to_ints(st)
    
    def get_floats(self):
        return np.hstack(str_to_multi_floats(str(self)))
    
    def get_multi_floats(self):
        return str_to_multi_floats(str(self))
    
    def get_array(self):
        return np.array(str_to_multi_floats(str(self)))


# In[23]:

if __name__ == "__main__":
    c1 = CffStr({'floats': np.array([-5.1, 2.2, 1.3]), 'precision': 2})
    c2 = CffStr({'ints': [1, 2, 3]})
    c3 = CffStr((-121, 34122, 140))
    c4 = CffStr({'cffs': [c1, c2, c3]})
    print(c1.get_floats())
    print(c2.get_floats())
    print(c3.get_ints())
    print(c4.get_array())
    


# In[27]:

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
        if type(initOpts) == unicode:
            initOpts = str(initOpts)
        if type(initOpts) == str:
            temp = initOpts.split('~')
            key = CffStr(temp[0])
            value = CffStr(temp[1])
        elif type(initOpts) == tuple: # constructed via Spark deserialization
            key = CffStr(initOpts[0])
            value = CffStr(initOpts[1])
        elif type(initOpts) == dict:
            if initOpts.has_key('csv_row'):
                ncoords = initOpts.get('ncoords', 3)
                temp = csv_row_to_array(initOpts['csv_row'])
                initOpts['coords'] = temp[:ncoords]
                initOpts['floats'] = temp[ncoords:]
            key = CffStr(initOpts['coords'])
            value = CffStr(initOpts)
        kv = tuple((key, value))
        assert(len(kv)==2)
        return super(Voxel, cls).__new__(cls, tuple(kv))
    
    # creates a new voxel from float data
    def convertData(self, floats, precision, coords = None):
        if coords is None:
            coords = self.get_coords()
        newVoxel = Voxel({'coords': self.get_coords(), 'floats': floats, 'precision': precision})
        return newVoxel
    
    def get_coords(self):
        coords = tuple(self[0].get_ints())
        return coords
    
    def get_floats(self):
        return self[1].get_floats()

    def get_multi_floats(self):
        return self[1].get_multi_floats()

    def get_array(self):
        return self[1].get_array()
    
    def bare_copy(self):
        return (str(self[0]), str(self[1]))
    
    def to_string(self): # used for writing to flat files
        return str(self[0])+'~'+str(self[1])
    
    def to_csv_string(self, delimiter=','): # used for writing to flat files
        return delimiter.join([str(v) for v in np.hstack([self.get_coords(), self.get_floats()])])


# In[28]:

if __name__ == "__main__":
    m = Voxel({'coords': (1, 2, 3), 'floats': np.array([1.12, 3.3, -4.5]), 'precision': 2})
    print(m.get_coords())
    print(m.get_floats())
    bc = m.bare_copy()
    print(bc)
    m_clone = Voxel(bc)
    print(m, m_clone)
    print(m.to_csv_string())


# In[7]:

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
    np.set_printoptions(precision=2)
    print(rawvoxes[50])
    print("")
    print(voxes[50].toCsvString())
    print("")
    print(Voxel(voxes[50].bareCopy()).toCsvString())
    print("")
    print(Voxel(voxes[50].toString()).toCsvString())
    print("")
    keys = [k for (k, v) in voxes]
    print(keys)


# # Testing Import

# In[32]:

if __name__ == "__main__":
    import numpy.random as npr
    from donuts.spark.classes import Voxel
    from StringIO import StringIO
    si = StringIO()
    # define functions used in testing
    nvox = 60
    def gen_vox():
        coords = npr.randint(0, 2, 3)
        data = npr.randn(20)
        return np.hstack([coords, data])
    # simulate text file
    rawvoxes= np.array([gen_vox() for ii in range(nvox)])
    np.savetxt(si, rawvoxes)
    rawdata = si.getvalue().strip().split('\n')
    voxes = [Voxel({'intRes': 3, 'csv_row': rawdat}) for rawdat in rawdata]
    np.set_printoptions(precision=2)
    print(rawvoxes[50])
    print("")
    print(voxes[50].toCsvString())
    print("")
    print(Voxel(voxes[50].bareCopy()).toCsvString())
    print("")
    print(Voxel(voxes[50].toString()).toCsvString())
    print("")
    keys = [k for (k, v) in voxes]
    print(sorted(keys))[0:20]


# # Testing in Spark

# In[33]:

if __name__ == "__main__" and 'sc' in vars():
    import donuts.spark.classes as dc
    def f(x):
        return Voxel({'intRes': 3, 'csv_row': x})
    raw_rdd = sc.parallelize(rawdata, 2)
    voxes = raw_rdd.map(f).collect()
    def v2c(v):
        return str(v)
    def c2c(c1, c2):
        return c1 + '~~' + c2
    combovoxes = raw_rdd.map(f).combineByKey(v2c, c2c, c2c).collect()
    print(combovoxes[0][0])
    print(combovoxes[0][1].split('~~'))


# In[8]:




# In[ ]:




# In[ ]:



