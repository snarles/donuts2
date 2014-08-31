# Run this file before running any simulations
# Usage: python setup_simulations.py [data_path]
# all simulation data will be saved in data path

import sys
datapath = sys.argv[1]
f = open('datapath.txt','w')
f.write(datapath)
f.close()



