sdata = (120, 120, 69, 150)
prefixes = ['/home/ubuntu/predator/8631_5_1_pfile/coil_images/8631_5_coil', \
            '/home/ubuntu/predator/8631_11_1_pfile/coil_images/8631_11_coil']
savedirs = ['/home/ubuntu/chris1/', '/home/ubuntu/chris2/']
suffix = '_ec.nii.gz'
import numpy as np
import nibabel as nib
fname = prefixes[dataset]+ str(ind) + suffix
rawdata = nib.load(fname).get_data()
