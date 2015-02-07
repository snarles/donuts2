
# Single-voxel noise estimation with multiple coils and directions


    f = open('../spark/sparkDonuts.py', 'r')
    exec(f.read())
    f.close()

# Introduction

## Model
 * Consider a single voxel, with measurements in directions $d = 1, ... ,n_d$
and coils $c = 1, ... , n_c$.
 * The B=0 case is treated as a direction (the 'zero' direction)
 * The coils have correlated noise, $\epsilon_dc$
 * The noise vector for a given direction $\epsilon_d$ has covariance $\Omega$,
and $\epsilon_d$ are independent for $d = 1,..., n_d$
 * Each coil also has a specific sensitivity $s_c \in [0,1]$
 * Each direction has an 'ideal signal' $\mu_d$
 * Define the separate coil measurements as $$y_{dc} = (s_c\mu_d +
\epsilon_{dc})^2$$
 * Define the combined-coil measurement as $$Y_d = \sum_{c} y_{dc}$$
 * There might be replicate measurements for a given direction.
    For example, if $d=1$ is the B0 direction, and if 10 B0 measurements were
taken, we would write them as
    $$Y_1^{(1)}, Y_1^{(2)}, ..., Y_1^{(10)}$$
Reference: Aviv et al (2014), Wright and Wald (1997)

## Problem

We wish to estimate $\sigma^2_d = Var(Y_d)$ for a given direction.

## Option 1: Combined-coil

The first option is to make use of multiple **combined-coil** estimates for the
same direction.
That is,
$$\hat{Var}(Y_d) = \frac{1}{m-1} \sum_{i=1}^m (Y_d^{(m)} - \bar{Y}_d)^2$$
This is clearly an unbiased estimated.

## Option 2: Multiple-coil

The second option is to estimate the covariances of the *separate coils*, and
then make use of the fact that since

$$Y_d = \sum_c y_{dc}$$

that

$$Var(Y_d) = \sum_{c, c'} Cov(y_{dc}, y_{dc'})$$

But how can we estimate the covariance matrix  $\Sigma_d = (Cov(y_{dc},
y_{dc'}))_{cc'}$?

We can do so by combining information from **multiple directions** and
**multiple coils**.

## Estimating the covariance for multiple coils

Let us write the coil measurements in a $n_d \times n_c$ matrix $M$
$$
M = \begin{bmatrix}
y_{11} & ...& y_{1 c} & ... & y_{1 n_c} \\
...& ... & ... & ...& ... \\
y_{d1} & ...& y_{dc} & ... & y_{d n_c} \\
... & ... & ...& ... & ... \\
y_{n_d 1} & ...& y_{n_d c} & ... & y_{n_d n_c}\end{bmatrix}
$$

The first step is to estimate $E(y_{dc})$ .
To do so, note that $$E[y_{dc}] = s_c^2 \mu_d^2 + \sigma^2_d$$
where $\sigma^2_d = Var(\epsilon_{dc})$ (assumed constant for a given
direction).

Therefore
$$
E(M) =
\begin{bmatrix}
\mu_1^2 s_1^2 + \sigma^2_1 & ...& \mu_1^2 s_c^2 + \sigma^2_1 & ... & \mu_1^2
s_{n_c}^2  + \sigma^2_1\\
...& ... & ... & ...& ... \\
\mu_d^2 s_1^2  + \sigma^2_d & ...& \mu_d^2 s_c^2 +\sigma^2_d & ... & \mu_d^2
s_{n_c}^2 +\sigma^2_d\\
... & ... & ...& ... & ... \\
\mu_{n_d}^2 s_1^2 +\sigma^2_{n_d} & ...& \mu_{n_d}^2 s_c^2+\sigma^2_{n_d} & ...
& \mu_{n_d}^2 s_{n_c}^2+\sigma^2_{n_d}
\end{bmatrix}
$$
**which is a rank two matrix.**

That is,
$$
E(M) =
\begin{bmatrix}\sigma^2_1 \\ ... \\ \sigma^2_d \end{bmatrix} \begin{bmatrix}1 &
... & 1\end{bmatrix} +
\begin{bmatrix}\mu_1^2 \\ ... \\ \mu_{n_d}^2\end{bmatrix} \begin{bmatrix}s_1^2 &
... & s_{n_c}^2\end{bmatrix}
$$

### Therefore, estimate E(M) by the rank-two approximation of M.

Let $\hat{M}$ denote this rank-two estimate, with entries $m_{dc}$

Then estimate $\hat{Cov}(y_{dc}, y_{dc'}) = (y_{dc} - m_{dc})(y_{dc'} -
m_{dc'})$.

**Note:** This is not an unbiased estimate, but we can work out a correction
factor.

# Simulation


    np.set_printoptions(precision = 1, linewidth = 120)
    
    # problem parameters
    n_c = 32                             # number of coils
    n_d = 100                            # number of directions
    mu_d = np.abs(npr.normal(2, 2, n_d)) # ideal signals
    s_c = np.abs(npr.normal(2, 1, n_c))  # sensitivities
    s_c = s_c/max(s_c)                   # so s in [0,1]
    sigma_d = 0.3 * np.ones(n_d)         # noise standard deviations: use the same for all directions
    coil_independence = 1000             # controls degree of independence (larger = more independent)
    
    # generate the covariance matrix for noise
    temp = npr.randn(n_c, coil_independence)
    omega = np.corrcoef(temp)
    L_omega = np.linalg.cholesky(omega)

Compute ground truth for direction 1


    nreps_est = 100000
    reps = np.zeros(nreps_est)
    d = 0
    for rep_no in range(nreps_est):
        temp = np.zeros(n_c)
        # generate noise
        eps = np.squeeze(np.dot(L_omega, npr.randn(n_c)))
        for c in range(n_c):
            temp[c] = (s_c[c]*mu_d[d] + sigma_d[d] * eps[c])**2
        reps[rep_no] = sum(temp)
    
    var_gt = 1.0/(nreps_est-1) * sum((reps - np.mean(reps))**2)
    print("Ground truth variance: " + str(var_gt))

    Ground truth variance: 56.1419906373


## Generate data


    nreps = 2  # number of replicate datasets
    # generate the individual coil signals
    y_dc = [None] * nreps
    for rep_no in range(nreps):
        y_dc[rep_no] = np.zeros((n_d, n_c))
        for d in range(n_d):
            # generate noise
            eps = np.squeeze(np.dot(L_omega, npr.randn(n_c)))
            for c in range(n_c):
                y_dc[rep_no][d, c] = (s_c[c]*mu_d[d] + sigma_d[d] * eps[c])**2
    
    # generate the single-coil signals
    Y_d = [None] * nreps
    for rep_no in range(nreps):
        Y_d[rep_no] = np.zeros(n_d)
        for d in range(n_d):
            Y_d[rep_no][d] = sum(y_dc[rep_no][d, :])

## Comparison

* **Option 1: Replicate measurements:** Use 10 replicate measurements
* **Option 2: Rank 2 trick:** Only use one replicate, but use separate coils and
all directions
   1. Assume coil independence
   2. No assumption


    # Option 1
    
    nreps_o1 = nreps
    reps = np.array([Y_d[rep_no][0] for rep_no in range(nreps_o1)])
    var_hat_1 = 1.0/(nreps_o1-1) * sum((reps - np.mean(reps))**2)
    
    # Option 2
    
    M = y_dc[0]
    yd0 = y_dc[0][0, :]
    # compute M_hat
    u, s, v = nla.svd(M, full_matrices = False)
    Mhat = np.dot(np.dot(u[:, 0:1], np.diag(s[0:1])), v[0:1,:])
    Mhat_d0 = Mhat[0, :]
    # compute cov_hat
    cov_hat = np.zeros((n_c, n_c))
    for c1 in range(n_c):
        for c2 in range(n_c):
            cov_hat[c1, c2] = (yd0[c1] - Mhat_d0[c1])*(yd0[c2] - Mhat_d0[c2])
    # estimate variance Y_0
    var_hat_2ind = sum((yd0 - Mhat[0, :])**2) # use independence assumption
    var_hat_2 = sum(cov_hat.ravel()) # no assumption
    vals = np.array([var_hat_1, var_hat_2ind, var_hat_2])
    errs = (vals - var_gt)**2
    
    print("Ground truth variance\t\t: " + str("%.5f" % var_gt)+"\n")
    
    titles = ["Option 1 (" + str(nreps_o1) + " replicates)\t\t: ", "Opt. 2 - Independence\t\t: ", "Opt. 2 - No assumption\t\t: "]
    obs = [nreps_o1, 1, 1]
    for ii in range(3):
        print(titles[ii] + str("%.5f" % vals[ii]))
        print "Squared error           \t:",
        print str("%.5f" % errs[ii]),
        if (errs[ii] == min(errs)):
            print " (BEST)\n"
        else:
            print("\n")

    Ground truth variance		: 56.14199
    
    Option 1 (2 replicates)		: 21.96150
    Squared error           	: 1168.30599 
    
    Opt. 2 - Independence		: 77.94411
    Squared error           	: 475.33239 
    
    Opt. 2 - No assumption		: 42.94079
    Squared error           	: 174.27160  (BEST)
    


##Many runs


    # Number of runs
    nruns = 10
    record_vals = np.zeros((nruns, 3))
    record_errs = np.zeros((nruns, 3))
    # Use parameters from before
    mu_d = np.abs(npr.normal(2, 2, n_d)) # ideal signals
    s_c = np.abs(npr.normal(2, 1, n_c))  # sensitivities
    s_c = s_c/max(s_c)                   # so s in [0,1]
    sigma_d = 0.3 * np.ones(n_d)         # noise standard deviations: use the same for all directions
    coil_independence = 1000             # controls degree of independence (larger = more independent)
    temp = npr.randn(n_c, coil_independence)
    omega = np.corrcoef(temp)
    L_omega = np.linalg.cholesky(omega)
    reps = np.zeros(nreps_est)
    # ground truth
    reps = np.zeros(nreps_est)
    d = 0
    for rep_no in range(nreps_est):
        temp = np.zeros(n_c)
        # generate noise
        eps = np.squeeze(np.dot(L_omega, npr.randn(n_c)))
        for c in range(n_c):
            temp[c] = (s_c[c]*mu_d[d] + sigma_d[d] * eps[c])**2
        reps[rep_no] = sum(temp)
    var_gt = 1.0/(nreps_est-1) * sum((reps - np.mean(reps))**2)
    
    for rep_no in range(nreps_est):
        temp = np.zeros(n_c)
        # generate noise
        eps = np.squeeze(np.dot(L_omega, npr.randn(n_c)))
        for c in range(n_c):
            temp[c] = (s_c[c]*mu_d[d] + sigma_d[d] * eps[c])**2
        reps[rep_no] = sum(temp)
    for run_no in range(nruns):
        # generate the individual coil signals
        y_dc = [None] * nreps
        for rep_no in range(nreps):
            y_dc[rep_no] = np.zeros((n_d, n_c))
            for d in range(n_d):
                # generate noise
                eps = np.squeeze(np.dot(L_omega, npr.randn(n_c)))
                for c in range(n_c):
                    y_dc[rep_no][d, c] = (s_c[c]*mu_d[d] + sigma_d[d] * eps[c])**2
        # generate the single-coil signals
        Y_d = [None] * nreps
        for rep_no in range(nreps):
            Y_d[rep_no] = np.zeros(n_d)
            for d in range(n_d):
                Y_d[rep_no][d] = sum(y_dc[rep_no][d, :])
        # Option 1
        reps = np.array([Y_d[rep_no][0] for rep_no in range(nreps_o1)])
        var_hat_1 = 1.0/(nreps_o1-1) * sum((reps - np.mean(reps))**2)
        # Option 2
        M = y_dc[0]
        yd0 = y_dc[0][0, :]
        # compute M_hat
        u, s, v = nla.svd(M, full_matrices = False)
        Mhat = np.dot(np.dot(u[:, 0:1], np.diag(s[0:1])), v[0:1,:])
        Mhat_d0 = Mhat[0, :]
        # compute cov_hat
        cov_hat = np.zeros((n_c, n_c))
        for c1 in range(n_c):
            for c2 in range(n_c):
                cov_hat[c1, c2] = (yd0[c1] - Mhat_d0[c1])*(yd0[c2] - Mhat_d0[c2])
        # estimate variance Y_0
        var_hat_2ind = sum((yd0 - Mhat[0, :])**2) # use independence assumption
        var_hat_2 = sum(cov_hat.ravel()) # no assumption
        vals = np.array([var_hat_1, var_hat_2ind, var_hat_2])
        errs = (vals - var_gt)**2
        record_vals[run_no, :] = vals
        record_errs[run_no, :] = errs


    mean_vals = np.mean(record_vals, axis = 0)
    mean_errs = np.mean(record_errs, axis = 0)
    
    print("Ground truth variance\t\t: " + str("%.5f" % var_gt)+"\n")
    
    for ii in range(3):
        print(titles[ii] + str("%.5f" % mean_vals[ii]) + " mean")
        print "Squared error           \t:",
        print str("%.5f" % mean_errs[ii]),
        if (errs[ii] == min(errs)):
            print " (BEST)\n"
        else:
            print("\n")

    Ground truth variance		: 2.80041
    
    Option 1 (2 replicates)		: 1.75653 mean
    Squared error           	: 5.03842 
    
    Opt. 2 - Independence		: 2.93054 mean
    Squared error           	: 1.55281  (BEST)
    
    Opt. 2 - No assumption		: 2.52144 mean
    Squared error           	: 2.09259 
    


##Further work

Assuming correlation structure is shared, pool correlation estimates across
directions and voxels
