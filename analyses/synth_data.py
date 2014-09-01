# run setup_simulations.py first
# run this file inside the directory



# sample command:
#  python synth_data.py b:1000 n:50 true_k:3 kappa:multi
#   generates 50 synthetic voxels using bvecs for b=1000, each with 3 true directions and multiple kappa
# LIST OF OPTIONS
#  name       - str          - name the synthetic data
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
import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import dipy.data as dpd
import time
today = time.strftime("%d-%m-%Y")


# default parameters
name = ''
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
    if cmd[0]=='name':
        name = cmd[1]
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

# automatically generate a name
auto_name = False
if name=='':
    auto_name = True
    f = open(datapath + 'index.txt','r')
    txt = f.read().split('\n')
    f.close()
    simlist = []
    for t in txt:
        t = t.split(' ')
        nom = ''
        is_data = False
        for t2 in t:
            t3 = t2.split(':')
            if t3[0]=='name':
                nom = t3[1]
            if t3[0]=='type':
                if (t3[1]=='raw') or (t3[1]=='synth'):
                    is_data = True
        if is_data:
            simlist = simlist + [nom]
print simlist
counter = 0
for nom in simlist:
    if nom[:9]=='synthdata':
        no = int(nom.replace('synthdata',''))
        counter = max(counter,no)
name = 'synthdata'+str(counter+1)

# write into the index
desc_string = ''
desc_string = desc_string + 'name:'+name+' bvals:bvals'+str(bval)
desc_string = desc_string + ' type:synth'
desc_string = desc_string + ' true_sigma:'+str(true_sigma)
desc_string = desc_string + ' true_k'+str(true_k)
desc_string = desc_string + ' date:'+today+'\n'

f = open(datapath+'index.txt','a')
f.write(desc_string)
f.close()

# print the parameter values
print 'path:'+datapath+name+'.npy'
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

# load bvecs
bvecs = np.load(datapath+'bvecs'+str(bval)+'.npy')

# commence with the simulation

    
