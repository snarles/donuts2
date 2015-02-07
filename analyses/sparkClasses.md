
# Development of classes for Spark data analysis


    # Load some numpy data


    def readN(filename, n):
        with open(filename, 'r') as myfile:
            head = [next(myfile).strip() for x in xrange(n)]
        return head
    
    def csvrow2array(st):
        return np.array([float(s) for s in st.replace(',',' ').replace('  ',' ').split(' ')])
    
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
    
    def str2array(st):
        pts = st.split('|')
        arr = np.array([str2ints(pt) for pt in pts]).T
        return arr


    csv1 = readN('/root/data/old_data.csv', 1000)


    cff1 = readN('/root/data/chris2_comb.cff', 1000)

# Define a voxel class


    import numpy as np
    
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
    
    class Voxel:
        'A class representing a single 3-dimensional voxel.  \
        Attributes are coordinates (key) and data. Optionally: cached data.\
        Internally, it uses characters to compress integer-valued data.'
        cached = 0.0
        dataString = ''
        key = ''
        intRes = 10000
        
        def convertInts(self, ints):
            self.dataString = ints2str(ints)
        
        def __init__(self, floats=None, string=None, ints=None, csvInts=None, csvFloats=None, intRes=10000):
            self.intRes = intRes
            if (csvFloats is not None):
                floats = csvrow2array(csvFloats)
            if (csvInts is not None):
                ints = np.array(csvrow2array(csvInts), dtype=int)
            if (floats is not None):
                self.convertInts(np.hstack([floats[0:3], np.array(floats[3:] * self.intRes)]))
            if (ints is not None):
                self.convertInts(ints)
            if (string is not None):
                self.dataString = string
            ints = str2ints(self.dataString)
            coords = (ints[0], ints[1], ints[2])
            self.key = ints2str(coords)
            
        def getCoords(self):
            ints = str2ints(self.key)
            coords = (ints[0], ints[1], ints[2])
            return coords
        
        def getKey(self):
            return self.key
        
        def toString(self):
            return self.dataString
        
        def getIntData(self):
            return np.array(str2ints(self.dataString), dtype=int)
        
        def getFloatData(self):
            ints = np.array(str2ints(self.dataString), dtype=int)
            return np.hstack([ints[0:3], np.array(ints[3:]/float(self.intRes))])
        
        def getArrayData(self):
            return str2Array(self.dataString)
        
        def setCache(self, val):
            self.cached = val
            
        def getCache(self):
            return self.cached

## subclasses


    


    #vox = Voxel(csvInts = csv1[500])
    vox = Voxel(string = cff1[500])


    vox.getCoords()




    (0, 7, 17)




    vox.getIntData()




    array([   0,    7,   17,    1, 4046, 5710,  409, 1079,  922,  833, 1242,
           1001, 1522,  844,  700,  955, 2173,  888, 1202,  966, 1167,  881,
            392, 1126,  856, 1125,  779,  632,  729, 1108, 1877, 1632,  930,
           1844,  762,  933, 1366,  599, 1395,  929,  984,  939,  854,  695,
            836,  640, 1387,  856,  610,  563,  756, 1130, 2061, 1580,  802,
           1037, 1582,  996,  986,  562,  617, 1686, 1068, 1275,  915,  813,
            926,  884, 1330,  817, 1494,  575, 1080,  869, 1616,  732, 1648,
            823,  639,  760, 1276, 1331,  899,  756,  759, 1658, 1058, 1592,
            571,  858,  849,  867, 1329, 1274,  530,  814, 1013, 1414,  647,
           1087,  509, 1255, 1708,  673, 1344, 1090,  465,  531,  704, 2220,
           1323, 1219,  961, 1122, 1508,  903,  606,  880,  953, 1090,  620,
            899,  656,  662,  770,  442, 3337, 1285,  769,  914, 1290, 1163,
            795, 2086, 2051,  757, 1961,  885,  651, 1388, 1425, 1188,  469,
           1286, 1162, 1462, 2285,  879,  534, 1259,  881,  908, 1141,  645])




    vox.toString()




    '!(2"M{w`{I%{R,{z+{7*{8.{i,{,1{s*{C({g+{X9{.*{o.{A+{c-{x*{h%{A-{O*{O-{N){\\({#){*-{=5{n3{-+{?5{M){K+{B0{1\'{\\0{N+{>+{u+{H*{M({b*{;({+0{F*{O\'{g\'{8){E-{S7{r2{S){s,{P2{U,{\'+{w\'{7\'{n3{c,{o/{0+{0*{$+{;*{k/{g*{(1{W\'{D-{!*{\\2{w){-3{=*{.({*){I/{1/{h*{z){E){H3{G,{e2{_\'{@*{Q*{H*{Z/{f/{/&{q*{%,{80{a({2-{(&{\\.{v3{y({L/{u-{+&{0&{r({k9{]/{`.{R+{^-{K1{e+{$\'{c*{g+{V-{+\'{q*{z({;({A){S%{sF{(/{:){R+{//{?-{t){l8{17{h){F6{h*{l({60{G0{l.{3&{4/{;-{s1{7:{D*{f&{u.{z*{h+{)-{^({0'




    


    


    


    


    


    


    


    


    


    


    

# Surrogate for SparkContext /RDD


    class FakeRDD:
        partitions = []
        def __init__(self, partitions):
            self.partitions = partitions
            
        def map(self, func):
            newpartitions = []
            for partition in self.partitions:
                newpartition = [func(element) for element in partition]
                newpartitions.append(newpartition)
            return FakeRDD(newpartitions)
        
        def flatMap(self, func):
            newpartitions = []
            for partition in self.partitions:
                newpartition = []
                for element in partition:
                    newpartition = newpartition + func(element)
                newpartitions.append(newpartition)
            return FakeRDD(newpartitions)        
        
        def mapPartitions(self, func):
            newpartitions = []
            for partition in self.partitions:
                newpartitions.append(func(iter(partition)))
            return FakeRDD(newpartitions)   
        
        def reduce(self, func):
            a_list = self.collect()
            n_items = len(a_list)
            if n_items < 2:
                return []
            ans = func(a_list[0], a_list[1])
            for ind in range(2, n_items):
                ans = func(ans, a_list[ind])
            return ans
        
        def collect(self):
            ans = []
            for partition in self.partitions:
                ans = ans + partition
            return ans
        
        def cache(self):
            return
        
    class FakeSparkContext:
        name = ''
        def __init__(self, name = 'FakeContext'):
            self.name = name
            
        def textFile(self, tf, n_parts):
            ff = open(tf, 'r')
            a_list = ff.read().split('\n')
            ff.close()
            return self.parallelize(a_list, n_parts)
        
        def parallellize(self, a_list, n_parts):
            n_items = len(a_list)
            sz_part = n_items/n_parts + 1
            count = 0
            partitions = []
            for ind in range(n_parts):
                newcount = min(count + sz_part, n_items)
                newpartition = a_list[count:newcount]
                partitions.append(newpartition)
                count = newcount
            return FakeRDD(partitions)


    a_list = [1,5,4,6,8,8]
    sc = FakeSparkContext()
    rdd = sc.parallellize(a_list, 3)
    from operator import add
    rdd.reduce(add)




    32




    rdd.collect()




    [1, 5, 4, 6, 8, 8]




    iter([1,2,3])




    <listiterator at 0x7f193ef48050>




    


    


    


    


    


    


    


    
