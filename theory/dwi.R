####
## Diffusion Imaging Simulation
####

library(magrittr)
library(rgl)
library(Rcpp)

xyz <- read.table('theory/4352.xyz', header = FALSE, sep = " ")[, -1] %>% as.matrix
nxyz <- dim(xyz)[1]

subsphere <- function(a, thres, rand = TRUE) {
  if (rand) {
    a <- a[sample(dim(a)[1], dim(a)[1]), ]
  }
  xmat <- a[1, , drop = FALSE]
  for (i in 2:dim(a)[1]) {
    d <- colSums((t(xmat) - a[i, ])^2)
    if (min(d) > thres) {
      xmat <- rbind(xmat, a[i, ])
    }
  }
  xmat
}


cppFunction('NumericVector subIndices(NumericMatrix a, double thres) {
  int nrow = a.nrow();
  NumericVector out(nrow);
  out[0] = 0;
  int count = 1;
  for (int i = 1; i < nrow; i++) {
    for (int j = 1; j < count; j ++) {
      double norm = 0;
      int s = out[j];
      for (int k = 0; k < 3; k++) {
        norm += (a(i, k) - a(s, k)) * (a(i, k) - a(s, k));
      }
      if (norm < thres) {
        out[count] = i;
        count += 1;
      }
    }
  }
  return out;
}'
)

c <- subIndices(xyz, 0.1)

metasub <- function(a, thres, nits = 5) {
  best <- 0
  ans <- numeric()
  for (i in 1:nits) {
    xmat <- subsphere(a, thres)
    if (dim(xmat)[1] > best) {
      ans <- xmat
    }
  }
  ans
}

s1 <- metasub(xyz, 0.045, 30)
dim(s1)

s1 <- s1[1:150, ]
plot3d(s1)

s2 <- subsphere(xyz, 0.01)
s2 <- s2[1:750, ]
dim(s2)
plot3d(s2)


