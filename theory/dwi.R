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
library(transport)

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

#plot(ouSub(rnorm(10000), .99), type = "l")

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

## ** ADMM algorithm **
f2 <- function(x) sum(x^2)
lambda_nnls <- function(X, y, lambda, maxits = 5) {
  l1p <- 0
  for (i in 1:maxits) {
    b <- nnls(rbind(X, l1p), c(y, 0))$x
    if (sum(b) == 0) return(list(b = b, lambda = 0))
    l1p <- sqrt(lambda/sum(b))
  }
  list(b = b, lambda = l1p^2 * sum(b))
}
soft <- function(x, tt) pmax(0, x - tt)
admm_iterate <- function(X, Y, lambda, nu, rho, B, FF, E, W,
                         mc.cores = 1, ...) {
  # Update B
  R <- Y - E - FF - W
  B <- do.call(cbind,
               mclapply(1:dim(R)[2],
                        function(i)
                          lambda_nnls(X, R[, i], lambda/rho)$b, mc.cores = mc.cores))
  XB <- X %*% B
  # Update FF
  R <- Y - XB - E - W
  res <- svd(R)
  d <- res$d
  objective_t <- function(tt) {
    nu * sum(soft(d, tt))^2 + (rho/2) * sum(pmin(tt, d)^2)
  }
  ts <- seq(0, max(d), max(d)/1000)
  vals <- sapply(ts, objective_t)
  tt <- ts[order(vals)[1]]
  FF <- res$u %*% diag(soft(d, tt)) %*% t(res$v)
  # Update E
  E <- rho/(2 + rho) * (Y - XB - FF - W)
  # Update W
  W <- W - (Y - XB - FF - E)
  list(X = X, Y = Y, lambda = lambda, nu = nu, rho = rho,
       B = B, FF = FF, E = E, W = W, mc.cores = mc.cores)
}
admm_nuclear <- function(X, Y, lambda, nu, rho, 
                         B = 0 * zeros(dim(X)[2], dim(Y)[2]),
                         FF = 0 * Y, E = Y, W = 0 * Y,
                         mc.cores = 1, maxits = 10, ...) {
  objective <- function(X, Y, lambda, nu, rho, B, FF, E, W, ...) {
    f2(Y - X %*% B - FF) + nu * nnorm(FF)^2 + lambda * sum(abs(B))
  }
  pars <- list(X = X, Y = Y, lambda = lambda, nu = nu, rho = rho,
               B = B, FF = FF, E = E, W = W, mc.cores = mc.cores)
  objs <- numeric(maxits)
  feas <- numeric(maxits)
  for (i in 1:maxits) {
    t1 <- proc.time()
    pars <- do.call(admm_iterate, pars)
    objs[i] <- do.call(objective, pars)
    feas[i] <- with(pars, f2(Y - X %*% B - E - FF))
    proc.time() - t1
  }
  c(pars, list(objs = objs, feas = feas, objective = objective))
}

## ** Computing EMD distance **

## Computes N x M matrix of arc distance between v1[N, 3] and v2[M, 3]
arcdist <- function(v1, v2) {
  M <- abs(nmlzr(v1) %*% t(nmlzr(v2)))
  M[M > 1] <- 1; acos(M)
}

arc_emd <- function(v1, w1, v2, w2) {
  if (sum(w1) * sum(w2) == 0) return(pi)
  v1 <- v1[w1 > 0, , drop = FALSE]
  w1 <- w1[w1 > 0]
  v2 <- v2[w2 > 0, , drop = FALSE]
  w2 <- w2[w2 > 0]
  w1 <- w1/sum(w1)
  w2 <- w2/sum(w2)
  M <- arcdist(v1, v2)
  res <- suppressWarnings(transport(w1, w2, M))
  sum(M[as.matrix(res[, 1:2])] * res[, 3])
}

