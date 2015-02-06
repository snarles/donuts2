sdata = (120, 120, 69, 150)
prefixes = ['/home/ubuntu/predator/8631_5_1_pfile/coil_images/8631_5_coil', \
            '/home/ubuntu/predator/8631_11_1_pfile/coil_images/8631_11_coil']
savedirs = ['/home/ubuntu/chris1/', '/home/ubuntu/chris2/']
suffix = '_ec.nii.gz'
import numpy as np
import nibabel as nib
fname = prefixes[dataset]+ str(ind) + suffix
outname = savedirs[dataset] + 'coil' + str(ind)
rawdata = nib.load(fname).get_data()
rdr = rawdata.ravel()**2
coords = np.vstack(np.unravel_index(range(120*120*69), (120, 120, 69))).T
newdata = np.hstack([coords, ind * np.ones(993600, 1), np.array(rdr.reshape((993600, 150))*10000, dtype=int)])

def int2str(z):
    if (z < 120):
        return chr(z)
    else:
        resid = int(z % 120)
        z = int(z-resid)/120
        return int2str(z)+chr(120)+chr(resid)
    
def ints2str(zs):
    return ''.join(int2str(z) for z in zs)

strs = [ints2str(nn) for nn in newdata]
f = open(outname, 'w')
f.write('\n'.join(strs))
f.close()
