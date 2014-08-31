# run setup_simulations.py first
# run this file inside the directory



# sample command:
#  python synth_data.py b:1000 n:50 true_k:3 kappa:multi
#   generates 50 synthetic voxels using bvecs for b=1000, each with 3 true directions and multiple kappa
# LIST OF OPTIONS
#  n          - int          - number of voxels
#  b          - int          - b value, one of [1000,2000,3000]
#  true_k     - int          - number of true directions
#  weighting  - str          - how to weight the directions, ['equal','dirichlet']
#  true_sigma - float        -  noise parameter
#  scaling    - float        -  scale the data by this amount
#  kappa      - float/str    - if 'multi', use multiple kappa, otherwise, set the kappa

f = open('datapath.txt')
datapath = f.read()
f.close()

import sys
sa = sys.argv

# default parameters
bval = 1000
true_k = 1
true_kappa = 1.5
true_sigma = 0.1
scaling = 1.0
multi_kappa = False
weighting = 'equal'
n = 100

# parse the system arguments
for cmd in sa:
    cmd = cmd.split(':')
    if cmd[0]=='b':
        bval = int(cmd[1])
    if cmd[0]=='n':
        n = int(cmd[1])
    if cmd[0]=='true_k':
        true_k = int(cmd[1])
    if cmd[0]=='kappa':
        if cmd[1]=='multi':
            multi_kappa = True
        else:
            true_kappa = float(cmd[1])
    if cmd[0]=='scaling':
        scaling = float(cmd[1])
    if cmd[0]=='weighting':
        weighting = cmd[1]
    if cmd[0]=='true_sigma':
        true_sigma = float(cmd[1])

# print the parameter values
print 'path:'+datapath
print 'n:'+str(n)
print 'bval:'+str(bval)
print 'true_k:' + str(true_k)
print 'true_sigma' + str(true_sigma)
if multi_kappa:
    print 'multi_kappa:True'
else:
    print 'true_kappa:'+str(true_kappa)
if scaling != 1.0:
    print 'scaling:'+str(scaling)
if true_k > 1:
    print 'weighting:'+weighting

    
