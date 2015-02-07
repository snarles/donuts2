
# Misc functions for spark


    import cvxopt as cvx
    import numpy as np
    import numpy.random as npr
    import scipy as sp
    import scipy.optimize as spo

    /usr/local/lib/python2.7/dist-packages/setuptools-7.0-py2.7.egg/pkg_resources.py:1045: UserWarning: /home/snarles/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
      warnings.warn(msg, UserWarning)



    cvx.solvers.options['show_progress'] = False


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
    
    def arcdist(xx,yy):
        """ Computes pairwise arc-distance matrix"""
        dm = np.absolute(np.dot(xx,yy.T))
        dm[dm > .99999999999999999] = .99999999999999999
        dd = np.arccos(dm)
        return dd
    
    def divsum(x):
        return x/sum(x)



    k1 = 5
    k2 = 3
    pp = 3
    x1 = normalize_rows(npr.normal(0, 1, (k1, pp)))
    x2 = normalize_rows(npr.normal(0, 1, (k2, pp)))
    w1 = divsum(npr.exponential(1, k1))
    w2 = divsum(npr.exponential(1, k2))


    dm = arcdist(x1, x2)
    dd = dm.ravel()
    k1 = np.shape(x1)[0]
    k2 = np.shape(x2)[0]
    a1 = np.kron(np.ones((1,k2)), np.eye(k1))
    a2 = np.kron(np.eye(k2), np.ones((1,k1)))
    aeq = np.vstack([a1, a2])
    beq = np.hstack([w1, w2])



    np.set_printoptions(linewidth = 120)
    aeq




    array([[ 1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.]])




    np.shape(aeq)




    (8, 15)




    np.shape(dd)




    (15,)




    def arc_emd(x1, w1, x2, w2):
        dm = arcdist(x1, x2)
        dd = dm.ravel()
        k1 = np.shape(x1)[0]
        k2 = np.shape(x2)[0]
        a1 = np.kron(np.ones((1,k2)), np.eye(k1))
        a2 = np.kron(np.eye(k2), np.ones((1,k1)))
        aeq = np.vstack([a1, a2])
        beq = np.hstack([w1, w2])
        cvec = cvx.matrix(dd)
        Gmat = cvx.matrix(-np.eye(len(dd)))
        hvec = cvx.matrix(np.zeros(len(dd)))
        Amat = cvx.matrix(aeq)
        bvec = cvx.matrix(beq)
        sol = cvx.solvers.lp(cvec, Gmat, hvec, Amat, bvec)
        return sol


    arc_emd(x1, w1, x2, w2)


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-27-450214c3e9ec> in <module>()
    ----> 1 arc_emd(x1, w1, x2, w2)
    

    <ipython-input-26-288830a562bf> in arc_emd(x1, w1, x2, w2)
         13     Amat = cvx.matrix(aeq)
         14     bvec = cvx.matrix(beq)
    ---> 15     sol = cvx.solvers.lp(cvec, Gmat, hvec, Amat, bvec)
         16     return sol


    /usr/local/lib/python2.7/dist-packages/cvxopt-1.1.7-py2.7-linux-x86_64.egg/cvxopt/coneprog.pyc in lp(c, G, h, A, b, solver, primalstart, dualstart)
       3006 
       3007     return conelp(c, G, h, {'l': m, 'q': [], 's': []}, A,  b, primalstart,
    -> 3008         dualstart)
       3009 
       3010 


    /usr/local/lib/python2.7/dist-packages/cvxopt-1.1.7-py2.7-linux-x86_64.egg/cvxopt/coneprog.pyc in conelp(c, G, h, dims, A, b, primalstart, dualstart, kktsolver, xnewcopy, xdot, xaxpy, xscal, ynewcopy, ydot, yaxpy, yscal)
        681         try: f = kktsolver(W)
        682         except ArithmeticError:
    --> 683             raise ValueError("Rank(A) < p or Rank([G; A]) < n")
        684 
        685     if primalstart is None:


    ValueError: Rank(A) < p or Rank([G; A]) < n



    
