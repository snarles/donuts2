####
## Diffusion Imaging Simulation
####
## Various functions
####

library(magrittr)
library(rgl)
library(Rcpp)
library(pracma)

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

ou <- function(n, p, theta = 0.5) {
  wmat <- matrix(0, n, 2*n)
  wmat <- (theta ^ (row(wmat) - col(wmat) + n)) * ((row(wmat)  + n + 1)> col(wmat))
  w <- randn(2 * n, p)
  x <- wmat %*% w
}
x <- ou(10000, 3, 0.999)
plot3d(x[9000:10000, ], type = "l")

####
## BVecs and candidates
####

set.seed(1)
s1 <- metasub(xyz, 0.0565, 100) %*% rand3()
dim(s1)
#plot3d(s1)

set.seed(2)
s2 <- metasub(xyz, 0.009, 10)
dim(s2)
#plot3d(s2)




