

    import os
    import numpy as np
    import numpy.random as npr
    import subprocess
    import donuts.spark.classes as dc
    import nibabel as nib
    import time


    os.chdir('/root/ephemeral-hdfs/bin')
    sz = (20, 20, 20)
    parts = 30


    s3names = ['s3://rawpredator/chris1/8631_5_coil' + str(i) + '_ec.nii.gz' for i in range(0, 33)] + \
              ['s3://rawpredator/chris2/8631_11_coil' + str(i) + '_ec.nii.gz' for i in range(0, 33)]
    innames = ['8631_5_coil' + str(i) + '_ec.nii.gz' for i in range(0, 33)] + \
              ['8631_11_coil' + str(i) + '_ec.nii.gz' for i in range(0, 33)]
    tempnames = ['temp1_coil' + str(i) + '.txt' for i in range(0, 33)] + \
                ['temp2_coil' + str(i) + '.txt' for i in range(0, 33)]
    outnames = ['chris1_coil' + str(i) + '.pickle' for i in range(0, 33)] + \
               ['chris2_coil' + str(i) + '.pickle' for i in range(0, 33)]
    s3outnames = ['s3://chris1data/chris1_coil' + str(i) + '.pickle' for i in range(0, 33)] + \
                 ['s3://chris2data/chris2_coil' + str(i) + '.pickle' for i in range(0, 33)]


    ind = 0
    s3name = s3names[ind]
    inname = innames[ind]
    tempname = tempnames[ind]
    outname = outnames[ind]
    s3outname = s3outnames[ind]
    print inname, outname

    8631_5_coil0_ec.nii.gz chris1_coil0.pickle



    print('Downloading...')
    t1 = time.time()
    os.system('aws s3 cp '+s3name + ' .')
    td = time.time() - t1
    print(td)

    Downloading...
    5.59779906273



    print('Loading into python...')
    t1 = time.time()
    rawdata = nib.load(inname).get_data()
    tl = time.time() - t1
    print(tl)

    Loading into python...
    8.32722496986



    print('Converting to flat file...')
    t1 = time.time()
    dc.convscript(rawdata, tempname, (10, 10, 10))
    tc = time.time() - t1
    print(tc)

    Converting to flat file...
    Wrote to file...
    Copied to hadoop... temp/temp1_coil0.txt
    Cleaning up...
    210.891757011



    print('Pickling...')
    t1 = time.time()
    dc.VoxelPartition(textf = 'temp/'+tempname, cont = sc, parts = parts).save_as_pickle_file(outname)
    ts = time.time() - t1
    print(ts)

    Pickling...
    153.128750086



    print('Transferring...')
    t1 = time.time()
    os.system('./hadoop fs -get ' + outname + ' ' + outname)
    os.system('aws s3 cp --recursive ' + outname + ' ' + s3outname)
    tt = time.time() - t1
    print(tt)

    Transferring...
    7.02462387085



    print('Cleaning up...')
    t1 = time.clock()
    os.system('rm ' + inname)
    os.system('rm ' + tempname)
    os.system('rm -r ' + outname)
    os.system('./hadoop fs -rmr ' + tempname)
    os.system('./hadoop fs -rmr temp/' + tempname)
    os.system('./hadoop fs -rmr '+outname)
    tu = time.clock() - t1
    print(tu)

    Cleaning up...
    0.58


# Checking the result


    ind = npr.randint(0, 66)
    s3name = s3names[ind]
    inname = innames[ind]
    tempname = tempnames[ind]
    outname = outnames[ind]
    s3outname = s3outnames[ind]

    8631_11_coil22_ec.nii.gz chris2_coil22.pickle



    print inname, outname
    os.system('aws s3 cp ' + s3name + ' ' + inname)
    t1 = time.time()
    os.system('aws s3 cp --recursive ' + s3outname + ' ' + outname)
    os.system('./hadoop fs -mkdir ' + outname)
    os.system('./hadoop fs -put ' + outname + '/* ' + outname + '/')
    print(time.time() - t1)
    rawdata = nib.load(inname).get_data()

    7.60544514656



    t1 = time.time()
    tups = dc.VoxelPartition(picklef = outname, cont=sc).rdd.takeSample(False, 5)
    print(time.time() - t1)

    3.66230893135



    for tup in tups:
        dims = np.shape(tup[1])
        x0 = npr.randint(0, dims[0])
        x1 = npr.randint(0, dims[1])
        x2 = npr.randint(0, dims[2])
        coords = tup[0]
        inds = npr.randint(0, 150, 10)
        print(coords)
        print(zip(rawdata[coords[0]+x0, coords[1]+x1, coords[2]+x2, inds],tup[1][x0, x1, x2, inds]))

    (60, 20, 20)
    [(0.6768285, 0.67676), (0.87980491, 0.87988), (0.71758568, 0.71777), (0.70602375, 0.70605), (0.69754267, 0.69775), (0.56752139, 0.56738), (0.65407282, 0.6543), (1.2776331, 1.2773), (0.79057264, 0.79053), (1.8217716, 1.8213)]
    (80, 0, 40)
    [(0.40794775, 0.40796), (0.51400071, 0.51416), (0.50792915, 0.50781), (0.55661494, 0.55664), (0.45164883, 0.45166), (0.44984168, 0.44995), (0.66701722, 0.66699), (0.46970704, 0.46973), (0.55682003, 0.55664), (0.53826493, 0.53809)]
    (100, 0, 0)
    [(0.39065871, 0.39062), (0.32688323, 0.3269), (0.41088822, 0.41089), (0.42792785, 0.42798), (0.38797721, 0.38794), (0.50285441, 0.50293), (0.42013773, 0.42017), (0.42832246, 0.42822), (0.43048403, 0.43042), (0.47979614, 0.47974)]
    (60, 60, 60)
    [(0.40942541, 0.40942), (0.48160049, 0.48169), (0.60075563, 0.60059), (0.98844814, 0.98828), (0.39641222, 0.39648), (0.5739826, 0.57422), (0.50791746, 0.50781), (1.0493406, 1.0498), (1.2623806, 1.2627), (0.52434015, 0.52441)]
    (60, 100, 20)
    [(0.46359003, 0.46362), (0.48723188, 0.4873), (3.130914, 3.1309), (2.7382114, 2.7383), (0.6531049, 0.65332), (0.58853626, 0.58838), (0.41484222, 0.41479), (0.55699533, 0.55713), (0.46605211, 0.46606), (0.54403067, 0.54395)]


# Automate it


    for ind in range(0, 66):
        s3name = s3names[ind]
        inname = innames[ind]
        tempname = tempnames[ind]
        outname = outnames[ind]
        s3outname = s3outnames[ind]
        print inname, outname
    
        print('Downloading...')
        t1 = time.time()
        os.system('aws s3 cp '+s3name + ' .')
        td = time.time() - t1
        print(td)
    
        print('Loading into python...')
        t1 = time.time()
        rawdata = nib.load(inname).get_data()
        tl = time.time() - t1
        print(tl)
    
        print('Converting to flat file...')
        t1 = time.time()
        dc.convscript(rawdata, tempname, (10, 10, 10))
        tc = time.time() - t1
        print(tc)
    
        print('Pickling...')
        t1 = time.time()
        dc.VoxelPartition(textf = 'temp/'+tempname, cont = sc, parts = parts).save_as_pickle_file(outname)
        ts = time.time() - t1
        print(ts)
    
        print('Transferring...')
        t1 = time.time()
        os.system('./hadoop fs -get ' + outname + ' ' + outname)
        os.system('aws s3 cp --recursive ' + outname + ' ' + s3outname)
        tt = time.time() - t1
        print(tt)
    
        print('Cleaning up...')
        t1 = time.clock()
        os.system('rm ' + inname)
        os.system('rm ' + tempname)
        os.system('rm -r ' + outname)
        os.system('./hadoop fs -rmr ' + tempname)
        os.system('./hadoop fs -rmr temp/' + tempname)
        os.system('./hadoop fs -rmr '+outname)
        tu = time.clock() - t1
        print(tu)

    8631_5_coil0_ec.nii.gz chris1_coil0.pickle
    Downloading...
    4.82283115387
    Loading into python...
    8.49957609177


    
