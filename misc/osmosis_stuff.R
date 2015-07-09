####
## Reading diffusion data from HCP
####

library(oro.nifti)
library(magrittr)
library(rgl)
library(AnalyzeFMRI)
library(pracma)
source("theory/dwi.R")

plots <- FALSE
gimage <- function(a) image(fliplr(t(a)), col = gray(0:20/20))

extract_vox <- function(V, vx) {
  dm <- dim(V)
  n <- dim(vx)[1]
  ans <- matrix(0, n, dm[4])
  for (i in 1:n) {
    ans[i, ] <- V[vx[i, 1], vx[i, 2], vx[i, 3], ]
  }
  ans
}

rank_k_approx <- function(resid, k = 1) {
  if (k == 0) return(0 * resid)
  res <- svd(resid, nu = k, nv = k)
  res$u %*% (res$d[1:k] * t(res$v))
}

emds <- function(B1, B2, mc.cores = 3) {
  unlist(mclapply(1:dim(B1)[2], function(i) arc_emd(pts, B1[-1, i], pts, B2[-1, i]), mc.cores = mc.cores))
}

zattach <- function(ll) {
  for (i in 1:length(ll)) {
    assign(names(ll)[i], ll[[i]], envir=globalenv())
  }
}


#ddir <- function(s = "") paste0("/home/snarles/hcp/10s/115320/", s)
ddir <- function(s = "") paste0("/home/snarles/predator/osmosis/", s)


#list.files(ddir("T1w/Diffusion"))
list.files(ddir())


## ** B-values **

set.seed(2)
pts <- metasub(xyz, 0.009, 10)
p <- dim(pts)[1]
if (plots) plot3d(pts)

#bvals <- read.table(ddir("T1w/Diffusion/bvals"), header = FALSE, sep = "") %>% as.numeric
#bvi <- round(bvals/1000)
#bvecs <- read.table(ddir("T1w/Diffusion/bvecs"), header = FALSE, sep = "") %>% 
bvecs1 <- (read.table(ddir("SUB1_b1000_1.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix)[11:160, ]
bvecs2 <- (read.table(ddir("SUB1_b1000_2.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix)[11:160, ]

if (plots) { plot3d(bvecs1, size = 5); points3d(1.01 * bvecs2, col = "red", size = 5) }

## ** nifti **
(wm1 <- readNIfTI(ddir("SUB1_wm_mask.nii.gz")))
(wm2 <- readNIfTI(ddir("SUB2_wm_mask.nii.gz")))

dim(wm1) # 81 106 76
dim(wm2) # 81 106 76

## Determine ROIs

wms <- GaussSmoothArray(wm1 + wm2, ksize = 17)
dim(wms)

max(wms)
if (plots) gimage(wms[, , 40])
sum(wms > 1.9)

roi_inds <- which(wms > 1.9, arr.ind = TRUE)
if (plots) plot3d(roi_inds, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76))

nclust <- 50
set.seed(2)
clust <- kmeans(roi_inds, nclust, nstart = 10, iter.max = 30)$cluster
if (plots) {
  plot3d(0, 0, 0, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76))
  for (i in 1:nclust) {
    points3d(roi_inds[clust == i, , drop = FALSE], col = rainbow(nclust)[i])
  }  
}

## Get data

(diff1 <- readNIfTI(ddir("SUB1_b1000_1.nii.gz")))
temp <- extract_vox(diff1, roi_inds)
diff1r <- temp[, 11:160]
so1r <- temp[, 1:10]
rm(diff1); gc()

(diff2 <- readNIfTI(ddir("SUB2_b1000_1.nii.gz")))
temp <- extract_vox(diff2, roi_inds)
diff2r <- temp[, 11:160]
so2r <- temp[, 1:10]
rm(diff2); gc()

## Save data
save(list=c("diff1r", "so1r", "diff2r", "so2r", "bvecs1", "bvecs2",
            "roi_inds", "clust"),
     file="data/osmosis7804.RData")

####
##  Correct for noise floor
####

res <- lm(apply(so1r, 1, var) ~ rowMeans(so1r))
cf <- coef(res)
plot(rowMeans(so1r), apply(so1r, 1, var))
abline(a = cf[1], b = cf[2], col = "red", lwd = 2)


res <- lm(apply(so2r, 1, var) ~ rowMeans(so2r))
cf <- coef(res)
plot(rowMeans(so2r), apply(so2r, 1, var))
abline(a = cf[1], b = cf[2], col = "red", lwd = 2)

mean(apply(so1r, 1, var))
mean(rowSums(so1r^2))

help(pmax)

s2 <- 0
Y1 <- t(diff1r)#[, 1:10]
Y2 <- t(diff2r)#[, 1:10]
Yc1 <- sqrt(pmax(Y1^2 - s2, 0))
Yc2 <- sqrt(pmax(Y2^2 - s2, 0))

####
##  Fit NNLS
####

kappa <- 3
X1 <- cbind(1, stetan(bvecs1, pts, kappa))
t1 <- proc.time()
B1 <- multi_nnls(X1, Yc1, mc.cores = 3)
proc.time() - t1
mu1 <- sqrt((X1 %*% B1)^2 + s2)
resid1 <- Y1 - mu1

X2 <- cbind(1, stetan(bvecs2, pts, kappa))
B2 <- multi_nnls(X2, Yc2, mc.cores = 3)
mu2 <- sqrt((X2 %*% B2)^2 + s2)
resid2 <- Y2 - mu2

####
##  Check prediction error of NNLS
####

n <- dim(roi_inds)[1]
Yh1 <- sqrt((X1 %*% B2)^2 + s2)
Yh2 <- sqrt((X2 %*% B1)^2 + s2)
f2(Y1 - Yh1)/n ## 386109.4
f2(Y2 - Yh2)/n ## 386306
e_nnls <- emds(B1, B2, mc.cores = 3)
mean(e_nnls) ## 0.385



####
##  Look at the noise in a cluster
####

ii <- 1
plot3d(roi_inds, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76), col = "green")
points3d(roi_inds[clust == ii, , drop = FALSE], size = 4, col = "black")
points3d(roi_inds[clust == jj, , drop = FALSE], size = 4, col = "yellow")


layout(1)
for (ii in 1:20) {
  plot(svd(resid1[, clust == ii])$u[, 1], svd(resid2[, clust == ii])$u[, 1])
  title(paste(ii))
}

layout(matrix(1:2, 1, 2))
for (ii in 1:20) {
  plot(svd(Y1[, clust == ii])$u[, 1], svd(Y2[, clust == ii])$u[, 1])
  title("U")
  plot(svd(Y1[, clust == ii])$v[, 1], svd(Y2[, clust == ii])$v[, 1])
  title(paste(ii))
}

layout(matrix(1:2, 1, 2))
for (ii in 1:20) {
  plot(svd(resid1[, clust == ii])$u[, 1], svd(resid2[, clust == ii])$u[, 1])
  title("U")
  resid1[, clust == ii])$u[, 1]
  plot(svd(resid1[, clust == ii])$v[, 1], svd(resid2[, clust == ii])$v[, 1])
  title(paste(ii))
}



jj <- 11
plot(svd(resid1[, clust == ii])$u[, 1], svd(resid2[, clust == jj])$u[, 1])


ii <- 19
k <- 1
vv <- 3
plot3d(mu1[, clust==ii][, vv] * bvecs1)
points3d((mu1[, clust==ii][, vv] + 3 * rank_k_approx(resid1[, clust == ii], k)[, vv]) * bvecs1, col = "red")

####
##  Cross-residual prediction
####

ii <- 20
nf <- 0.05
errs1 <- numeric()
errs2 <- numeric()
n <- sum(clust == ii)
for (k in 1:10) {
  Yh1 <- X1 %*% B2[, clust == ii] + nf * rank_k_approx(resid2[, clust == ii], k)
  Yh2 <- X2 %*% B1[, clust == ii] + nf * rank_k_approx(resid1[, clust == ii], k)
  errs1[k] <- f2(Y1[, clust == ii] - Yh1)/n
  errs2[k] <- f2(Y2[, clust == ii] - Yh2)/n  
}
plot(errs1, type = "l")
plot(errs2, type = "l")



####
## Apply ADMM 
####

ii <- 20
lambda <- 0.1
nu <- 0.1
rho <- 1
mcc <- 3
adr1 <- admm_nuclear(X1, Yc1[, clust == ii],
                     lambda = lambda, nu = nu, rho = rho, mc.cores = mcc)
t1 <- proc.time()
adr2 <- admm_nuclear(X2, Yc2[, clust == ii],
                     lambda = lambda, nu = nu, rho = rho, mc.cores = mcc)
proc.time() - t1


## corellation between noise

layout(1)
plot(svd(adr1$FF)$u[, 1], svd(adr2$FF)$u[, 1])
title(paste(ii))

plot(adr1$FF, adr2$FF, pch = ".")
plot(rank_k_approx(adr1$FF, 1), rank_k_approx(adr2$FF, 1), pch = ".")
plot(rank_k_approx(adr1$FF, 2), rank_k_approx(adr2$FF, 2), pch = ".")

sapply(1:4, function(k) {
  cor(as.numeric(rank_k_approx(adr1$FF, k)), as.numeric(rank_k_approx(adr2$FF, k)))
})


k <- 1
vv <- 3
plot3d((adr1$X %*% adr1$B)[, vv] * bvecs1)
points3d((adr1$Y[, vv] + rank_k_approx(adr1$FF, k)[, vv]) * bvecs1, col = "red")

## prediction errors given added noise

n <- dim(adr1$Y)[2]
errs1 <- numeric()
errs2 <- numeric()
nf <- 0.05
for (k in 1:10) {
  Yh1 <- (adr1$X %*% adr2$B) + nf * rank_k_approx(adr2$FF, k)
  Yh2 <- (adr2$X %*% adr1$B) + nf * rank_k_approx(adr1$FF, k)
  errs1[k] <- f2(adr1$Y - Yh1)/n
  errs2[k] <- f2(adr2$Y - Yh2)/n
}
plot(errs1, type = "l")
plot(errs2, type = "l")

