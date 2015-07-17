####
##  Random 3d splines for diffusion simulation
####

source("theory/dwi.R")

library(expm)
library(pracma)
library(tensorA)
library(rgl)
library(prodlim)






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

elms <- ou(n=1000, p=3, theta = 0.99)
plot(elms[, 1])

n <- 1000
theta <- 0.99
eps <- .1

splinepath <- function(n, theta, eps, eps2) {
  elms <- ou(n, p=3, theta)
  #elms2 <- eps * elms
  elms2 <- eps * apply(elms, 2, cumsum)
  elms3 <- array(0, dim = c(3, 3, n))
  elms4 <- array(0, dim = c(3, 3, n))
  for (i in 1:3) elms4[i, i, ] <- 1
  elms3[1, 2, ] <- elms2[, 1]
  elms3[1, 3, ] <- elms2[, 2]
  elms3[2, 3, ] <- elms2[, 3]
  elms3[2, 1, ] <- -elms2[, 1]
  elms3[3, 1, ] <- -elms2[, 2]
  elms3[3, 2, ] <- -elms2[, 3]
  res <- sapply(1:n, function(i) expm::expm(elms3[, , i], method="Higham08")[1, ])
  path <- eps2 * apply(res, 1, cumsum)
  path <- (rand3() %*% (t(path) - path[floor(n/2), ])) + path[floor(n/2)]
  path
}

#path <- splinepath(1000, 0.99, 0.01, 0.01)
#plot3d(t(path), type = "l")
#apply(path, 2, max)

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
##  Use a unit square for the ROI
##  Populate it with fibers
####

Nres <- 40
roi <- array(0, dim = c(Nres, Nres, Nres, p))
nfibers <- 80
fiberid <- array(0, dim = c(Nres, Nres, Nres, nfibers))
npath <- 1000
nactual <- 0
pathmult <- 20
pathrad <- 0.15
for (ii in 1:nfibers) {
  w <- runif(1) # weight of the fiber
  path <- splinepath(npath, 0.99, 0.01, 0.01)
  center <- runif(3)
  for (jj in 1:pathmult) {
    path <- path - rowMeans(path) + center + pathrad * runif(3)    
    # make sure the path exits the box
    if (max(apply(path, 1, min)) < 0 && min(apply(path, 1, max)) > 1) {
      # find where the path intersects the roi
      discret_path <- round(path * Nres)[, -1]
      # unique rows
      vox <- unique(t(discret_path))
      # filter by inclusion in box
      vox <- vox[vox[, 1] > 0 & vox[, 1] <= Nres &
                   vox[, 2] > 0 & vox[, 2] <= Nres &
                   vox[, 3] > 0 & vox[, 3] <= Nres, , drop = FALSE]
      if (length(vox) > 6) {
        nactual <- nactual + 1
        # convert path to weight matrix
        diffs <- path[, -1] - path[, -npath]
        inds <- row.match(data.frame(t(discret_path)), data.frame(vox),
                          nomatch = 0)
        diff2 <- diffs[, inds != 0]
        inds2 <- inds[inds != 0]
        ip <- t(diff2) %*% t(pts)
        wd <- t(apply(abs(ip), 1, function(v) v == max(v)))
        for (i in 1:max(inds)) {
          roi[vox[i, 1], vox[i, 2], vox[i, 3], ] <-
            roi[vox[i, 1], vox[i, 2], vox[i, 3], ] + 
            w * colSums(wd[inds2 == i, , drop = FALSE])
          fiberid[vox[i, 1], vox[i, 2], vox[i, 3], ii] <- 1
        }    
      }    
    }
  }
}

## plot total weight of fibers in ROI

tot <- apply(roi, c(1, 2, 3), sum)
plot3d(which(tot > 0, arr.ind=TRUE), size = 1)
cols <- hsv(h = 1:nfibers/nfibers, s =1, v = 0.5)
for (i in 1:nfibers) {
  points3d(which(fiberid[, , , i] > 0, arr.ind=TRUE), size = 5, col = cols[i])
}

## make "little" voxels

library(tensorA)
dim(roi)
length(roi)
dim(roi)
troi <- as.tensor(roi, dims=c(x = Nres, y = Nres, z = Nres, u = p))
kappa <- 1
X <- stetan(bvecs, pts, kappa)
tX <- as.tensor(X, dims = c(g = nb, u = p))
res <- troi %e% tX
roi_mean <- as.array(res$x)

gimage(roi_mean[, , 20, 1])
gimage(roi_mean[, , 20, 2])
gimage(roi_mean[, , 20, 100])
gimage(roi_mean[, , 30, 100])

roi_inds <- which(tot > 0, arr.ind=TRUE)  
diff_vol <- extract_vox(roi_mean, roi_inds)
dim(diff_vol)

## try ICA

library(fastICA)
library(AnalyzeFMRI)
cm <- hsv(h = 0.5, s = 1, v = 1:100/100, alpha=1:100/100)
cm2 <- hsv(h = 0.9, s = 1, v = 1:100/100, alpha=1:100/100)
spplot <- function(x0, ...) {
  x <- abs(x0)
  cols <- ifelse(x0 > 0, 
                 cm[floor(99 * (x - min(x))/(max(x) - min(x))) + 1],
                 cm2[floor(99 * (x - min(x))/(max(x) - min(x))) + 1])
  plot3d(roi_inds, col = cols, ...)
}

bvplot <- function(x, ...) {
  cols <- cm[floor(99 * (x - min(x))/(max(x) - min(x))) + 1]
  plot3d(bvecs, col = cols, ...)
}

bvplot2 <- function(x, eps = 0.1, ...) {
  mod <- x/sd(x)
  cols <- cm[floor(99 * (x - min(x))/(max(x) - min(x))) + 1]
  plot3d(pts, col = "gray")  
  points3d((1 + eps * mod) * bvecs, col = cols, ...)
}

mean(diff_vol)
Y1 <- array(sqrt(rchisq(length(diff_vol), df=2, ncp=diff_vol^2)), dim = dim(diff_vol))

dim(Y1)


res1 <- ICAspat(t(Y1), n.comp=149)
dim(res1$time.series) # 150 100
dim(res1$spatial.components) # 100 1801

o <- order(-apply(res1$time.series, 2, var))
TS <- res1$time.series[, o]
SC <- res1$spatial.components[o, ]

spplot(SC[1, ], size = 9)
bvplot2(-TS[, 1], size = 20)
bvplot2(-TS[, 2], size = 20)
bvplot2(-TS[, 3], size = 20)
bvplot2(-TS[, 4], size = 20)
bvplot2(-TS[, 5], size = 20)
bvplot2(-TS[, 6], size = 20)
bvplot2(-TS[, 7], size = 20)
bvplot2(-TS[, 100], size = 20)

spplot(SC[2, ], size = 9)
spplot(SC[3, ], size = 9)
spplot(SC[100, ], size = 9)

bvplot2(X[, 1] - mean(X[, 1]), size = 20)

