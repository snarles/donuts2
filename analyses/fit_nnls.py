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
#  data       - str          - name of datafile WITHOUT extension (looked up in index)
#  name       - str          - optional : job name
#  grid       - int          - tells the script to use single kappa, specifies number of candidate directions [362,1442]
#  multi_reso - int          - resolution for multi-kappa
#  kfold      - int          - tells the script to use cross-validation, specifies number of folds
#             - str          - kfold:n to use n-fold cross validation (jackknife)
#  kappas     - (float x3)   - tells the script to use single kappa, (min,max,step) for kappas
#  l1ps       - (float x3)   - tells the script to use multi-kappa, (min,max,step) for k1 penalties
#  nunits     - int          - number of units (eg voxels) per job
#  out_cves   -              - tells the script to output cves per voxel per kappa/l1p
#  out_emds   -              - tells the script to output earthmover distance per kappa/l1p (synthetica data only)
#  print      -              - tells the script to output the progress

import sys
sa = sys.argv

f = open('datapath.txt')
datapath = f.read()
f.close()

f = open(datapath + 'index.txt','r')
index = f.read().split('\n')
f.close()

import os.path
import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import donuts.deconv.navigator as dnv
import time
import math
today = time.strftime("%d-%m-%Y")

# parameter ranges

allowed_grid = [362,1442]

# Default parameters
name = ''
datafile = 'cc1000'
multi_kappa = False
grid_size = 362
multi_reso = -1
kfold = 20
kappas = np.arange(1,2,.1)
l1ps = -1
out_cves = False
out_emds = False
print_opt = False
nunits = 100

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
        multi_kappa = True
        multi_res = int(sss[1])
    if sss[0]=='kfold':
        kfold = int(sss[1])
    if sss[0]=='kappas':
        ss2 = sss[1].replace('(','').replace(')','')
        kp = [float(ss3) for ss3 in ss2.split(',')]
        kappas = np.arange(kp[0],kp[1],kp[2])
    if sss[0]=='l1ps':
        ss2 = sss[1].replace('(','').replace(')','')
        kp = [float(ss3) for ss3 in ss2.split(',')]
        l1ps = np.arange(kp[0],kp[1],kp[2])
    if sss[0] == 'nunits':
        nunits = int(sss[1])
    if sss[0]=='out_cves':
        out_cves = True
    if sss[0]=='out_emds':
        out_emds = True
    if sss[0]=='print':
        print_opt == True

# Get the data and determine the total number of parts

data = np.load(datapath + datafile + '.npy')

# Look up the record in the index and get bvecs

record = dnv.query(index, 'name', datafile)[0]
bvecs = np.load(datapath + record['bvecs'] + '.npy')
n = np.shape(data)[0]
nparts = int(math.ceil(float(n)/nunits))

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

# now we know if the job exists; determine if the job needs to be run
if not job_exist:
    part = 0
    if name=='':
        # automatically generate the name
        counter = 0
        tokens = [ele.split(' ') for ele in index]
        for tok1 in tokens:
           for tok2 in tok1:
               tok3 = tok2.split(':')
               if (tok3[0]=='name') & (tok3[1][:3]=='fit'):
                   counter = max(int(tok3[1][3:]), counter)
        name = fit + str(counter+1)
    # create the job in the job index
    f = open('jobqueue.txt','a')
    f.write(cmdstring + '|' + name + '|' + str(part) + '\n')
    print 'Creating new job with name '+name
if job_exist:
    # determine the part number
    part = part+1

    print 'Continuing existing job: now part '+part

# Run the job

# Save the results

# Write the results in the index

# Check to see if this is the last part: if so, add the job to the analysis queue
