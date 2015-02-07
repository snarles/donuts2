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






dataset = 1
sdata = (120, 120, 69, 150)
inds_filt = [16, 31, 46, 61, 75, 90, 105, 120, 135, 149]
sdata0 = (sdata[0], sdata[1], sdata[2])
ndata = sdata[0]*sdata[1]*sdata[2]
pdata = sdata[3]
intres = 10000
prefixes = ['/home/ubuntu/predator/8631_5_1_pfile/coil_images/8631_5_coil', \
            '/home/ubuntu/predator/8631_11_1_pfile/coil_images/8631_11_coil']
savedirs = ['/home/ubuntu/tempdata/', '/home/ubuntu/tempdata/']
suffix = '_ec.nii.gz'
import numpy as np
import nibabel as nib
fname = prefixes[dataset]+ str(ind) + suffix
outname = savedirs[dataset] + 'B0_coil' + str(ind) + '.cff'




rawdata = nib.load(fname).get_data()
rdr = rawdata.ravel()**2
coords = np.vstack(np.unravel_index(range(ndata), sdata0)).T
newdata = np.hstack([coords, ind * np.ones((ndata, 1), dtype=int), \
                     np.array(rdr.reshape((ndata, pdata))[:, inds_filt]*intres, dtype=int)])
strs = [ints2str(nn) for nn in newdata]





strs = [ints2str(nn) for nn in newdata]
f = open(outname, 'w')
f.write('\n'.join(strs))
f.close()
