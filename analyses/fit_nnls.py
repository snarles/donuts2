# run setup_simulations.py first
# run this file inside the directory
# if prompted, run again (since one run may only scan part of the data)
#
# fits an nnls model to data
#
# sample command:
#  python fit_nnls.py data:cso1000 grid:362 kfold:20 kappas:(1,2,0.1) out_cves
#   fits nnls model with 362 candidate directions, using kappa from [1,1.1,...,2] using 20-fold cross validation, return cves
# another sample command:
#  python fit_nnls.py data:synthdata1 multi_reso:10 l1ps:(0.01,0.1,0.01) out_emds
#   fits nnls model using multi-kappa, using l1 penalties [0.01,0.02,...,0.1], return Eartmover Distances (only for synthetic data)
# 
# LIST OF OPTIONS
#  name       - str          - optional : job name
#  grid       - int          - tells the script to use single kappa, specifies number of candidate directions [362,1442]
#  multi_reso - int          - resolution for multi-kappa
#  kfold      - int          - tells the script to use cross-validation, specifies number of folds
#             - str          - kfold:n to use n-fold cross validation (jackknife)
#  kappas     - (float x3)   - tells the script to use single kappa, (min,max,step) for kappas
#  l1ps       - (float x3)   - tells the script to use multi-kappa, (min,max,step) for k1 penalties
#  out_cves   -              - tells the script to output cves per voxel per kappa/l1p
#  out_emds   -              - tells the script to output earthmover distance per kappa/l1p (synthetica data only)
#  print      -              - tells the script to output the progress

f = open('datapath.txt')
datapath = f.read()
f.close()


import os.path
import sys
sa = sys.argv
import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import dipy.data as dpd
import time
today = time.strftime("%d-%m-%Y")

# parameter ranges

allowed_grid = [362,1442]

# Default parameters
name = ''
multi_kappa = False
grid_size = 362
multi_reso = -1
kfold = 20
kappas = np.arange(1,2,.1)
l1ps = -1
out_cves = False
out_emds = False
print_opt = False

# parse input arguments
for ss in sa:
    sss = ss.split(':')
    if sss[0]=='name':
        name = sss[1]
    if sss[0]=='grid':
        multi_kappa = False
        xx = int(sss[1])
        if xx in allowed_grid:
            grid_size = xx
        else:
            print 'Grid size ' + sss[1]+' not allowed.'
            print 'Choose from: '+','.join(allowed_grid)
    if sss[0]=='multi_reso':


# Checks to see if the job has already been queued: if so, it will increment the part

cmdstring = ' '.join(sa)
f = open('jobqueue.txt','a')
f.close()
f = open('jobqueue.txt','r')
joblist = f.read().split('\n')
f.close()
joblist = [jj.split('|') for jj in joblist]

# has this command been run?
job_exist = False
part =-1
for jj in joblist:
    if jj[0]==cmdstring:
        job_exist = True
        name = jj[1]
        pt = int(jj[2])
        part = max(part,pt)
if not job_exist:
    if name=='':
    else:



# Run the job

# Save the results

# Write the results in the index

# Check to see if this is the last part: if so, add the job to the analysis queue
