# run setup_simulations.py first
# run this file inside the directory

# fits an nnls model to data

# sample command:
#  python fit_nnls.py b:1000 n:50 true_k:3 kappa:multi
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


