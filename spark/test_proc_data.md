
# Check that the data was converted correctly


    full_chris1 = ['/home/ubuntu/predator/8631_5_1_pfile/coil_images/'+ 'coil_comb_ec.nii.gz'] + \
            ['/home/ubuntu/predator/8631_5_1_pfile/coil_images/8631_5_coil' + str(i) + '_ec.nii.gz' for i in range(1,33)]
    outnames1 =   ['/home/ubuntu/chris1/chris1_comb.cff'] + \
            ['/home/ubuntu/chris1/chris1_coil' + str(i) + '.cff' for i in range(1,33)]
    
    full_chris2 = ['/home/ubuntu/predator/8631_11_1_pfile/coil_images/'+ 'coil_comb_ec.nii.gz'] + \
            ['/home/ubuntu/predator/8631_11_1_pfile/coil_images/8631_11_coil' + str(i) + '_ec.nii.gz' for i in range(1,33)]
    outnames2 =   ['/home/ubuntu/chris2/chris2_comb.cff'] + \
            ['/home/ubuntu/chris2/chris2_coil' + str(i) + '.cff' for i in range(1,33)]
    
    innames = full_chris1 + full_chris2
    outnames = outnames1 + outnames2
    from donuts.spark.classes import Voxel
    from donuts.spark.classes import CffStr
    import numpy.random as npr
    import numpy as np
    import nibabel as nib
    np.set_printoptions(precision = 4, linewidth=120)


    ind = npr.randint(0, len(outnames))
    #ind = npr.randint(24, 54)
    #ind = 33
    outname = outnames[ind]
    inname = innames[ind]
    print(outname, inname)
    f = open(outname, 'r')
    lines = f.read().strip().split('\n')
    f.close()
    rawdata = nib.load(inname).get_data()

    ('/home/ubuntu/chris2/chris2_coil6.cff', '/home/ubuntu/predator/8631_11_1_pfile/coil_images/8631_11_coil6_ec.nii.gz')



    nvox = len(lines)
    ii = npr.randint(0, nvox)
    print("Voxels: " + str(nvox))
    v = Voxel(lines[ii])
    coords = v.getCoords()
    print(coords)
    print("")
    print(rawdata[coords[0], coords[1], coords[2], :])
    print("")
    print(v.getData())

    Voxels: 993600
    (103.0, 94.0, 25, 39)
    
    [  8.3719e-02   1.5230e-01  -2.9512e-02   3.6186e-02   2.5387e-05   4.3389e-02   1.1908e-01   1.4586e-01   2.3072e-01
       1.3623e-01   3.1040e-02  -1.2804e-01  -2.9127e-02   4.7017e-02   1.3112e-03   2.8932e-02  -1.0213e-02  -1.5627e-03
       5.9975e-02   5.8925e-02   7.4881e-02  -1.9760e-02   4.0270e-03   6.1566e-02   5.6125e-02   6.8964e-02   9.5141e-03
       1.3331e-01   7.7408e-02   8.4386e-02   1.6174e-01   1.6497e-01  -6.9871e-02   1.3823e-01  -7.7626e-02   2.9389e-02
       1.0509e-01  -1.2511e-01   1.3506e-01  -2.0262e-02  -1.3794e-02   3.0532e-02   4.4525e-03   6.7714e-03   9.7101e-04
       1.1563e-01   1.6425e-01   2.1009e-02   2.6126e-02   1.1694e-01   1.0689e-01   3.8888e-02  -3.3478e-03   1.0568e-01
      -9.8237e-02   3.6134e-03   2.1605e-01   1.1310e-01   1.1308e-01   8.5849e-02   9.8554e-02   2.5406e-02   2.1631e-01
       9.6586e-02   6.6880e-02  -3.6540e-02   1.7579e-01   2.7678e-02   8.0977e-02   1.0726e-01   1.3802e-01   1.2232e-01
       5.0421e-02   1.2072e-01  -1.3077e-01   1.1449e-01   4.2102e-02   6.3940e-02   5.9284e-02  -9.2197e-03   1.0751e-01
       5.6172e-02   2.4511e-02  -3.2513e-02   5.6820e-02   5.9351e-02   1.7039e-01  -6.0997e-02   8.5146e-02  -4.6531e-02
       7.0436e-02  -5.8730e-02   5.2093e-02   1.4114e-01   1.0958e-01   7.1381e-04   4.8645e-03   1.2909e-01   1.0017e-01
      -1.1565e-01   7.8654e-02  -6.5548e-02   3.7367e-02   2.3093e-02   1.1240e-01   7.5223e-02  -2.5228e-02   9.8165e-02
      -3.0832e-02   4.4414e-02   6.1377e-02   1.5292e-02   6.3964e-02   8.0256e-02   1.2181e-01   1.3944e-01   4.8836e-02
       1.3131e-01   1.4892e-01   4.8205e-02  -3.0601e-03   5.4516e-02  -1.5370e-02  -9.1628e-02   4.3536e-02  -2.0070e-01
       1.5990e-01   2.8893e-03   2.1100e-01   2.0062e-02   4.2381e-02   6.9903e-02   8.0355e-02  -2.7160e-02   9.7927e-02
       7.8255e-02   1.9578e-01   5.3032e-02   1.1030e-01   1.0463e-01  -1.7995e-02   5.7364e-02   1.7758e-02   1.7838e-01
       7.9522e-02   1.7624e-02   1.6296e-02  -4.6420e-03  -3.7533e-02  -8.9765e-02]
    
    [ 0.0837  0.1522 -0.0295  0.0361  0.      0.0433  0.119   0.1458  0.2307  0.1362  0.031  -0.128  -0.0291  0.047   0.0013
      0.0289 -0.0102 -0.0015  0.0599  0.0589  0.0748 -0.0197  0.004   0.0615  0.0561  0.0689  0.0095  0.1333  0.0774  0.0843
      0.1617  0.1649 -0.0698  0.1382 -0.0776  0.0293  0.105  -0.1251  0.135  -0.0202 -0.0137  0.0305  0.0044  0.0067  0.0009
      0.1156  0.1642  0.021   0.0261  0.1169  0.1068  0.0388 -0.0033  0.1056 -0.0982  0.0036  0.216   0.113   0.113   0.0858
      0.0985  0.0254  0.2163  0.0965  0.0668 -0.0365  0.1757  0.0276  0.0809  0.1072  0.138   0.1223  0.0504  0.1207 -0.1307
      0.1144  0.0421  0.0639  0.0592 -0.0092  0.1075  0.0561  0.0245 -0.0325  0.0568  0.0593  0.1703 -0.0609  0.0851 -0.0465
      0.0704 -0.0587  0.052   0.1411  0.1095  0.0007  0.0048  0.129   0.1001 -0.1156  0.0786 -0.0655  0.0373  0.023   0.1124
      0.0752 -0.0252  0.0981 -0.0308  0.0444  0.0613  0.0152  0.0639  0.0802  0.1218  0.1394  0.0488  0.1313  0.1489  0.0482
     -0.003   0.0545 -0.0153 -0.0916  0.0435 -0.2006  0.1599  0.0028  0.211   0.02    0.0423  0.0699  0.0803 -0.0271  0.0979
      0.0782  0.1957  0.053   0.1102  0.1046 -0.0179  0.0573  0.0177  0.1783  0.0795  0.0176  0.0162 -0.0046 -0.0375 -0.0897]



    def vox2str(ind, ii, rawdata):
        dims = np.shape(rawdata)
        dim0 = tuple(dims[:3])
        coords = np.unravel_index(ii, dim0)
        v = rawdata[coords[0], coords[1], coords[2], :]
        return Voxel({'coords': list(coords) + [ind], 'intRes': 4, 'floats': v}).toString()


    
