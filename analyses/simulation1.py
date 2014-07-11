import numpy as np
import donuts.deconv.utils as du

import dipy.data as dpd
s1 = dpd.get_sphere('symmetric362')
s2 = s1.subdivide() # s2 has 1442 vertices
bvecs = s1.vertices
grid = s2.vertices


