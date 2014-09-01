# run setup_simulations.py first
# run this file inside the directory

# generates synthetic data using bvecs from real data

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

import sys
sa = sys.argv

f = open('datapath.txt')
datapath = f.read()
f.close()
f = open(datapath + 'index.txt','r')
index = f.read().split('\n')
f.close()

import numpy as np
import donuts.data as dnd
import donuts.deconv.utils as du
import donuts.deconv.navigator as dnv
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

if name=='':
    name = dnv.autonamer(index,'name','synthdata')

# print the parameter values
print 'path:'+datapath+name+'.npy'
print 'n:'+str(n)
print 'bval:'+str(bval)
print 'true_k:' + str(true_k)
print 'true_sigma:' + str(true_sigma)
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
nd = np.shape(bvecs)[0]
print 'number of directions:'+str(nd)

true_poss = np.zeros((true_k,3,n))
true_ws = np.zeros((true_k,n))

res = np.zeros((n,nd))
res0 = np.zeros((n,nd))
for ii in range(n):
    true_pos = du.normalize_rows(np.random.normal(0,1,(true_k,3)))
    if multi_kappa:
        for i in range(true_k):
            true_pos[i,] = true_pos[i,]*np.random.uniform(1,2)
    else:
        true_pos = np.sqrt(true_kappa) * true_pos
    true_poss[:,:,ii] = true_pos
    true_w = np.ones((true_k,1))/true_k
    true_ws[:,ii] = true_w
    if weighting=='dirichlet':
        true_w =np.random.dirichlet(true_w).reshape((-1,1))
    y0, y1 = du.simulate_signal_kappa(np.sqrt(true_kappa)*true_pos,true_w,bvecs,true_sigma)
    res[ii,] = np.squeeze(y1)
    res0[ii,] = np.squeeze(y0)

np.save(datapath + name+'_pos',true_poss)
np.save(datapath + name+'_w',true_ws)
np.save(datapath + name+'_y0',res0)
np.save(datapath + name, res)

# write into the index
desc_string = ''
desc_string = desc_string + 'name:'+name+' bvals:bvals'+str(bval)
desc_string = desc_string + ' type:synth'
desc_string = desc_string + ' true_sigma:'+str(true_sigma)
desc_string = desc_string + ' true_k:'+str(true_k)
desc_string = desc_string + ' date:'+today+'\n'

f = open(datapath+'index.txt','a')
f.write(desc_string)
f.close()
