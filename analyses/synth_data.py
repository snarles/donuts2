# run setup_simulations.py first
# run this file inside the directory



# sample command:
#  python synth_data.py b:1000 n:50 true_k:3 kappa:multi
#   generates 50 synthetic voxels using bvecs for b=1000, each with 3 true directions and multiple kappa


f = open('datapath.txt')
datapath = f.read()
f.close()

import sys
sa = sys.argv

# default parameters
bval = 1000
true_k = 1
true_kappa = 1.5
multi_kappa = False
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

# print the parameter values
print 'path:'+datapath
print 'n:'+str(n)
print 'bval:'+str(bval)
print 'true_k:' + str(true_k)
if multi_kappa:
    print 'multi_kappa:True'
else:
    print 'true_kappa:'+str(true_kappa)
