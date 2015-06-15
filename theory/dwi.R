####
## Diffusion Imaging Simulation
####
## Various functions
####

library(magrittr)
library(rgl)
library(Rcpp)
library(pracma)
library(nnls)
library(parallel)

## ** Functions for constructing spheres **

xyz <- read.table('theory/4352.xyz', header = FALSE, sep = " ")[, -1] %>% as.matrix

cppFunction('NumericVector subIndices(NumericMatrix a, double thres) {
  int nrow = a.nrow();
  NumericVector out(nrow);
  out[0] = 0;
  int count = 1;
  for (int i = 1; i < nrow; i++) {
    double minnorm = 4;
    for (int j = 0; j < count; j ++) {
      double norm = 0;
      int s = out[j];
      for (int k = 0; k < 3; k++) {
        norm += (a(i, k) - a(s, k)) * (a(i, k) - a(s, k));
      }
      if (norm < minnorm) {
        minnorm = norm;
      }
    }
    if (minnorm > thres) {
      out[count] = i;
      count += 1;
    }
  }
  return out;
}'
)

metasub <- function(a, thres, nits = 5) {
  best <- 0
  ans <- numeric()
  for (i in 1:nits) {
    a2 <- a[sample(dim(a)[1], dim(a)[1]), ]
    xi <- subIndices(a2, thres)
    xi <- xi[1:(which(xi == 0)[2] - 1)] + 1
    xmat <- a2[xi, ]
    if (dim(xmat)[1] > best) {
      best <- dim(xmat)[1]
      ans <- xmat
    }
  }
  ans
}

rand3 <- function() svd(randn(3))$u
#rand3() %>% {t(.) %*% .}

## normalize rows
nmlzr <- function(a) a/sqrt(rowSums(a^2))

## ** Functions for diffusion design matrix **

stetan <- function(bvecs, cands, kappa = 1) exp(-kappa * (bvecs %*% t(cands))^2)

## ** Generation of random parameters

## Ornstein-Uhlembeck process

cppFunction('NumericVector ouSub(NumericVector x, double theta) {
  int n = x.size();
  NumericVector out(n);
  double y = 0;
  for (int i = 0; i < n; i++) {
    double y = theta * y + x[i];
    out[i] = y;
  }
  return(out);
}'
)

plot(ouSub(rnorm(10000), .99), type = "l")

ou <- function(n, p = 1, theta = 0.5) {
  burnin <- floor(-7/log(theta))
  sdz <- sqrt(1/(1 -(theta^2)))
  x <- matrix(0, n, p)
  for (i in 1:p) {
    x[, i] <- ouSub(rnorm(n + burnin), theta)[burnin + (1:n)]/sdz
  }
  x
}

#theta <- 0.999
#x <- ou(10000, 3, theta)
#sd(x[, 1])
#sqrt(sum((theta^2) ^ (0:10000)))
#plot3d(nmlzr(x), type = "l")

## ** Optimization **

multi_nnls <- function(X, Y, mc.cores = 3) {
  v <- dim(Y)[2]
  bs <- mclapply(1:v, function(i) nnls(X, Y[, i])$x, mc.cores = mc.cores)
  do.call(cbind, bs)
}

nnorm <- function(E) {
  res <- svd(E, nu = 0, nv = 0)
  sum(res$d)
}

rank1approx <- function(E) {
  res <- svd(E, nu = 1, nv = 1)
  res$u %*% t(res$v)
}

nuclear_opt <- function(X, Y, lambda, maxits = 10, mc.cores = 3) {
  v <- dim(Y)[2]
  p <- dim(X)[2]
  n <- dim(X)[1]
  Z <- Y
  B <- multi_nnls(X, Z, mc.cores = mc.cores)
  objective <- function(B, Z) {
    c(sum((Z - X %*% B)^2), nnorm(Z - Y)/lambda)
  }
  objs <- objective(B, Z)
  for (it in 1:maxits) {
    # find the projection
    B <- multi_nnls(X, Z, mc.cores = mc.cores)
    Zh <- X %*% B
    resid <- Z - Zh
    alpha <- 1/it
    Z <- Y + (1-alpha) * (Z - Y)  + alpha * lambda * rank1approx(-resid)
    # objective
    objs <- rbind(objs, objective(B, Z))
  }
  list(E = Z - Y, B = B, objs = objs, objective = objective)
}

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
dim(pts)
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
E <- 1.1 * randn(nb, rk) %*% randn(rk, n)
nnorm(E)
Y <- Mu + 0.1 * randn(nb, n) + E

####
## Do the denoising
####

X <- stetan(bvecs, pts, kappa)
B_nnls <- multi_nnls(X, Y)
Mu_nnls <- X %*% B_nnls
(err_nnls <- sum((Mu - Mu_nnls)^2))

res_nn <- nuclear_opt(X, Y, 10.0, 100)
B_nn <- res_nn$B
Mu_nn <- X %*% B_nn
(err_nn <- sum((Mu - Mu_nn)^2))
