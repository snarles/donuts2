####
##  Random 3d splines for diffusion simulation
####

source("theory/dwi.R")

library(expm)
library(pracma)
library(tensorA)

## parallel matrix multiplication, A[, , i] * B[, , i]

pmm <- function(A, B) {
  n <- dim(A)[1]; k <- dim(A)[2]; p <- dim(B)[2]
  N <- dim(A)[3]
  ans <- array(0, dim=c(n, p, N))
  for (i in 1:n) {
    for (j in 1:p) {
      for (K in 1:k) {
        ans[i, j, ] <- ans[i, j, ] + A[i, K, ] * B[K, j, ]
      }
    }
  }
  ans
}

N <- 100000
A <- array(rnorm(3 * 3 * N), dim = c(3, 3, N))
B <- array(rnorm(3 * 3 * N), dim = c(3, 3, N))

t1 <- proc.time()
C <- lapply(1:N, function(i) {A[, , i] %*% B[, , i]})
proc.time() - t1
t1 <- proc.time()
C2 <- pmm(A, B)
proc.time() - t1

C[[1]]
C2[, , 1]

elms <- ou(n=1000, p=3, theta = 0.99)
plot(elms[, 1])

n <- 1000
theta <- 0.99
eps <- .1

splinepath <- function(n, theta, eps) {
  elms <- ou(n, p=3, theta)
  elms2 <- apply(elms, 2, cumsum)
  
  apply(elms, 1, function(v) {
    a <- matrix(0, 3, 3)
    a[upper.tri(a)] <- v
    a <- a - t(a)
    
  })
}
