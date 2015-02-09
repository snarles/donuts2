
# Processing data uisng Spark


    import donuts.spark.classes as dc
    import numpy as np
    from operator import add


    parts = 300
    textfiles = ['chris2/chris2_coil' + str(i) + '.cff' for i in range(1, 33)]


    sc




    <pyspark.context.SparkContext at 0x7f531d0a2690>




    
    # converts coil coords to regular coords, moving one coord over
    # and extract B0 values
    def coil_coords_conv(s): 
        inds_b0 = np.array([16, 31, 46, 61, 75, 90, 105, 120, 135, 149], dtype=int)
        try:
            ss = s.split('~')
            coords = dc.str2ints(ss[0])
            data = dc.CffStr(ss[1]).getFloats()
            newdata = data[inds_b0]
            newcoords = tuple(coords[:3])
            ns0 = dc.CffStr(newcoords)
            ns10 = dc.CffStr(coords)
            ns1 = dc.CffStr({'intRes': 4, 'floats': newdata})
            ans = (str(ns0), str(ns10)+'~'+str(ns1))
        except:
            ans= ["ERROR", s]
        return ans
    def v2c(v):
        return str(v)
    def c2c(v1, v2):
        vt = v1.split('~~') + v2.split('~~')
        vt = sorted(vt)
        return '~~'.join(vt)
    def str2multiVox(s):
        st = s.split('~~~')
        st2 = st[1].split('~~')
        vs = [dc.Voxel(st3) for st3 in st2]
        return vs
    def toS(v):
        return v[0]+'~~~'+v[1]
    
    def align_check(st):
        ans = []
        vs = str2multiVox(st)
        coords = [tuple(v.getCoords()[:3]) for v in str2multiVox(smpl)]
        return sum([coord != coords[0] for coord in coords])
    
    def combB0str(st):
        key = st.split('~~~')[0]
        vs = str2multiVox(st)
        st = key+'~'+'~'.join([v[1] for v in vs])
        return st


    smp = sc.textFile(textfiles[0], parts).takeSample(False, 10)
    coil_coords_conv(smp[0])




    ('"{:"{2#', 'HEAD"{:"{2#"~%}T|"{/|-{i\'{:){&%{R*{/I${u3{X')




    parts = 300
    combined_rdd = sc.textFile(','.join(textfiles), parts).map(coil_coords_conv).combineByKey(v2c, c2c, c2c)
    combined_rdd.map(toS).saveAsTextFile('out3.txt')


    #smp = sc.textFile('out3.txt', parts).takeSample(False, 10)


    sc.textFile('out3.txt', parts).map(align_check).reduce(add)




    0




    sc.textFile('out3.txt', parts).map(combB0str).saveAsTextFile('chris2_B0.cff')


    
