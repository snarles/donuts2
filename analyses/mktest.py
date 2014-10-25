# multi kappa test

import numpy as np
import numpy.linalg as nla
import donuts.deconv.utils as du 

# simulation part

bvecs = du.randvecsgap(1000,0)
true_kappas = np.array([1, 1.5])
true_pos = du.randvecsgap(2,.8)
true_kpos = np.sqrt(np.tile(true_kappas,(3,1)).T) * true_pos 
true_w = np.array([[.9,.1]]).T
sigma = 0.01
[y0,y] = du.simulate_signal_kappa(true_kpos,true_kappas,bvecs,sigma)

gsph = du.sph_lattice(10,4)
gsph = gsph[gsph[:,0] >= 0, :]
gkaps = np.sqrt(du.norms(gsph))
lamb = 0.1
xs = np.vstack([du.ste_tan_kappa(gsph,bvecs),lamb * gkaps])

[yh,beta,est_pos,est_w] = du.ls_est(y,xs,gsph)



