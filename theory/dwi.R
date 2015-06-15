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

dd <- dist(a2) %>% as.matrix
od <- dd[xi, -xi]
dim(od)
max(apply(od, 2, min))^2

set.seed(1)
s1 <- metasub(xyz, 0.0565, 100)
dim(s1)
plot3d(s1)

set.seed(2)
s2 <- metasub(xyz, 0.009, 10)
dim(s2)
plot3d(s2)




