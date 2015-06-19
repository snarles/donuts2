source("theory/dwi.R")

####
## BVecs and candidates
####

set.seed(1)
bvecs <- metasub(xyz, 0.0565, 100) %*% rand3()
dim(bvecs)
plot3d(bvecs)
nb <- dim(bvecs)[1]

set.seed(2)
pts <- metasub(xyz, 0.009, 10)
p <- dim(pts)[1]
plot3d(pts)

####
## Generate parameters
####

## number of fiber directions
kdir <- 3
## number of voxels
n <- 200
## kappa
kappa <- 3
## correlation between directions
theta <- 0.9
## correlation between weights
wcorr <- 0.99
temp <- lapply(1:kdir, function(i) nmlzr(ou(n, 3)))
dirs <- lapply(1:n, function(i) do.call(rbind, lapply(temp, function(v) v[i, ])))
ws <- abs(ou(n, kdir, wcorr)) %>% {./rowSums(.)}
## generate means
mus <- lapply(1:n, function(i) as.numeric(stetan(bvecs, dirs[[i]], kappa) %*% ws[i, ]))
Mu <- do.call(cbind, mus)
# low-rank noise
rk <- 1
E <- 0.1 * randn(nb, rk) %*% randn(rk, n)
nnorm(E)
Y <- Mu + 0.01 * randn(nb, n) + E

####
## Do the denoising
####

X <- stetan(bvecs, pts, kappa)
B_nnls <- multi_nnls(X, Y)
Mu_nnls <- X %*% B_nnls
(err_nnls <- sum((Mu - Mu_nnls)^2))

res_nn <- nuclear_opt(X, Y, 10.0, 10)
B_nn <- res_nn$B
Mu_nn <- X %*% B_nn
(err_nn <- sum((Mu - Mu_nn)^2))

res_admm <- admm_nuclear(X, Y, lambda = 0.1, nu = 0.2, rho = 2,
                         mc.cores=3, maxits = 10)
B_admm <- res_admm$B
Mu_admm <- X %*% B_admm
(err_admm <- sum((Mu - Mu_admm)^2))
