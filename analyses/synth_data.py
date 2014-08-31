# run setup_simulations.py first
# run this file inside the directory



# sample command:
#  python synth_data.py b:1000 n:50 true_k:3 kappa:multi
#   generates 50 synthetic voxels using the HCP parameters for b=1000


f = open('datapath.txt')
datapath = f.read()
f.close()

import sys

sa = sys.argv
# part the system arguments
