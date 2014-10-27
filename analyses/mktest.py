# multi kappa test

import numpy as np
import numpy.linalg as nla
import donuts.deconv.utils as du 
import dipy.data as dpd



# simulation part

s1 = dpd.get_sphere('symmetric362')
#bvecs = du.randvecsgap(1000,0)
bvecs = s1.vertices
true_kappas = np.array([1, 1.5])
true_pos = du.randvecsgap(2,.8)
true_kpos = np.sqrt(np.tile(true_kappas,(3,1)).T) * true_pos 
true_w = np.array([[1,0]]).T
sigma = 0.1


gsph = du.sph_lattice(30,4)
gsph = gsph[gsph[:,0] >= 0, :]
gkaps = np.sqrt(du.norms(gsph))
xs = np.vstack([du.ste_tan_kappa(gsph,bvecs),0 * gkaps])
[y0,y] = du.simulate_signal_kappa(true_kpos,true_w,bvecs,sigma)

lamb = 0.1
xs[-1,:] = lamb*gkaps
[yh,beta,est_pos,est_w] = du.ls_est(y,xs,gsph)



