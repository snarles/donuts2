
# coding: utf-8

# In[ ]:

def norms(x) :
    """ computes the norms of the rows of x """
    nms = np.sum(np.abs(x)**2,axis=-1)**(1./2)
    return nms

def normalize_rows(x):
    """ normalizes the rows of x """
    n = np.shape(x)[0]
    p = np.shape(x)[1]
    nms = norms(x).reshape(-1,1)
    x = np.multiply(x, 1./np.tile(nms,(1,p)))
    return x

def arcdist(xx, yy):
    """ Computes pairwise arc-distance matrix"""
    dm = np.absolute(np.dot(xx,yy.T))
    dm[dm > .99999999999999999] = .99999999999999999
    dd = np.arccos(dm)
    return dd

def divsum(x):
    return x/sum(x)

def arc_emd(x1, w1, x2, w2):
    x1 = x1[w1 > 0, :]
    w1 = w1[w1 > 0]
    x2 = x2[w2 > 0, :]
    w2 = w2[w2 > 0]
    w1 = divsum(w1)
    w2 = divsum(w2)
    arg1 = np.hstack([w1, 0*w2])
    arg2 = np.hstack([0*w1, w2])
    arg3 = arcdist(np.vstack([x1, x2]), np.vstack([x1, x2]))
    return emd(arg1, arg2, arg3)

