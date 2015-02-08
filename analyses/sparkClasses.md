
# Development of classes for Spark data analysis


    import donuts.spark.classes as ds


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
    cff1 = readN('/root/data/chris1_comb.cff', 1000)
    cff2 = readN('/root/data/chris2_comb.cff', 1000)


    comboset = cff1[:10] + cff2[:10]


    def f(x):
        lala = ds.Voxel(x)
        lala.setConversion(minVal = 0.0)
        return lala


    voxes = sc.parallelize(comboset,2).map(f).collect()


    def lister(a):
        return [a]
    
    def addList(a, b):
        return a + b


    voxes = sc.parallelize(comboset,2).map(f)


    cvoxes = voxes.combineByKey(lister, addList, addList)


    cvoxes_c = cvoxes.collect()


    len(cvoxes_c)




    9




    cvoxes_c[0]




    ('!!%',
     ['!!%"X{nX{3({<+{Z){H\'{D\'{)){2&{R*{X*{V\'{#({S){Z*{@%{1*{!({]({Q){H\'{"+{%&{.+{1&{j*{!*{v({w({\'+{/\'{.({g%{-\'{S){0({L({E&{/\'{H\'{j%{z){b\'{C({#&{R*{o({z,{@){,\'{h&{x*{8({3,{1({B,{w\'{C+{!({C({h${c+{Q%{7){w){\'\'{e+{\\&{/\'{z({\\({V*{b({l){j({m\'{^*{%&{R){V\'{L*{?\'{5({)*{.\'{.){3${k${6\'{8&{t/{7({@\'{S({K\'{g*{#){7&{l){"\'{1({)({=${h&{m*{B({(,{_*{h\'{R\'{y&{i%{`\'{B&{O){e\'{P&{$%{v){w${`1{c\'{t*{G&{z){H.{i.{^+{?({^*{<\'{H({.({5&{(&{^({s({@){I\'{C&{o\'{[\'{A${H%{_){3&{E%{W({V*{$&{7',
      '!!%""{7{(P{z({,%{r\'{w({x+{p\'{$\'{N#{D,{r({G&{m%{#&{+${U+{Y%{@%{W({$#{L({A&{2+{\\%{N&{[){6%{<*{E&{5%{"){G&{=){@\'{F+{4\'{=&{z%{-*{#({:${r\'{N\'{7%{l&{x){f+{<&{(%{d-{:%{\'%{C){G){g){A&{H&{E%{A%{\\({4%{n%{C\'{G\'{P${,\'{7%{Z.{I\'{k\'{l\'{d&{M({3\'{2({E${R${-\'{j%{E,{6&{i){n\'{=${z\'{z%{d%{n\'{/\'{g%{B%{.&{Z&{_0{:*{P&{X%{?#{q({E&{P\'{V\'{j%{[({V*{]&{g,{k%{a*{o%{p\'{s({p%{d({0&{U#{D-{$({m%{R\'{<\'{5({L*{F&{W#{h${8%{0${F${S${`\'{A){a&{Q*{c({V){E%{U${i){##{v%{p${K#{Y({0({,({H%{U&{w&{)'])




    sc.parallelize(comboset,2).map(f).


    v2=voxes[5].convertData(np.array([5.5, 1.1]), intRes = 10.0, minVal = -1.0)


    v2




    ('!!&', '!!&b6')




    v2.getData()




    array([ 5.5,  1.1])




    type('haha')




    str




    ds.Voxel(cff1[10])




    ('!', '!')




    list(['a','b'])




    ['a', 'b']




    [c for c in list(['aa'])[0]]




    ['a', 'a']



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
        
        def parallelize(self, a_list, n_parts):
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




    sc0 = FakeSparkContext()


    


    


    


    


    


    

# Test Spark stuff


    


    


    
