import numpy as np
import nibabel as nib
import donuts 
import os
import os.path as op

data_path = op.join(donuts.__path__[0], 'data')

def load_hcp_cso():
    """
    Load a 40-by-40-by-1 chunk of the HCP multi-b-value data from subject
    100307. 
    
    Returns
    -------
    data, bvecs, bvals
    
    """
    bvecs = np.loadtxt(op.join(data_path, '100307bvecs'))
    bvals = np.loadtxt(op.join(data_path, '100307bvals'))
    data = np.load(op.join(data_path, '100307small40thru80_100thru140_58.npy'))
    return data, bvecs, bvals

def load_hcp_cc():
    """
    Load corpus callosum from subject
    100307. 
    
    Returns
    -------
    data, bvecs, bvals
    
    """
    bvecs = np.loadtxt(op.join(data_path, '100307bvecs'))
    bvals = np.loadtxt(op.join(data_path, '100307bvals'))
    data = np.load(op.join(data_path, '100307_corpus_callosum.npy'))
    return data, bvecs, bvals
