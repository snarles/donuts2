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


emds <- function(B1, B2, mc.cores = 3) {
  unlist(mclapply(1:dim(B1)[2], function(i) arc_emd(pts, B1[-1, i], pts, B2[-1, i]), mc.cores = mc.cores))
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
bvi <- round(bvals/1000)
#bvecs <- read.table(ddir("T1w/Diffusion/bvecs"), header = FALSE, sep = "") %>% 
bvecs1 <- (read.table(ddir("SUB1_b1000_1.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix)[11:160, ]
bvecs2 <- (read.table(ddir("SUB1_b1000_2.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix)[11:160, ]

if (plots) plot3d(bvecs1); points3d(1.01 * bvecs2, col = "red")

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
sum(wms > 1.99)

roi_inds <- which(wms > 1.99, arr.ind = TRUE)
if (plots) plot3d(roi_inds, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76))

nclust <- 20
clust <- kmeans(roi_inds, nclust, nstart = 10)$cluster
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

####
##  Fit NNLS
####

kappa <- 3
X1 <- cbind(1, stetan(bvecs1, pts, kappa))
Y1 <- t(diff1r)#[, 1:10]
B1 <- multi_nnls(X1, Y1, mc.cores = 3)
mu1 <- X1 %*% B1
resid1 <- Y1 - mu1

X2 <- cbind(1, stetan(bvecs2, pts, kappa))
Y2 <- t(diff2r)#[, 1:10]
B2 <- multi_nnls(X2, Y2, mc.cores = 3)
mu2 <- X2 %*% B2
resid2 <- Y2 - mu2

####
##  Check prediction error of NNLS
####

n <- dim(roi_inds)[1]
Yh1 <- X1 %*% B2
Yh2 <- X2 %*% B1
f2(Y1 - Yh1)/n ## 386109.4
f2(Y2 - Yh2)/n ## 386306
e_nnls <- emds(B1, B2, mc.cores = 3)
mean(e_nnls) ## 0.385



####
##  Look at the noise in a cluster
####

layout(1)

layout(matrix(1:2, 1, 2))
plot(svd(resid1)$d)
plot(svd(resid2)$d)
ii <- ii + 1
for (ii in 1:nclust) {
  plot(svd(resid1[, clust == ii])$d)
  title(paste(ii))  
  plot(svd(resid2[, clust == ii])$d)
  title(paste(ii))  
}



plot3d(roi_inds, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76), col = "green")
points3d(roi_inds[clust == 11, , drop = FALSE], size = 4, col = "black")

plot(svd(resid1[, clust == 11])$u[, 1])
plot(svd(resid2[, clust == 11])$u[, 1])

layout(1)
for (ii in 1:20) {
  plot(svd(resid1[, clust == ii])$u[, 1], svd(resid2[, clust == ii])$u[, 1])
  title(paste(ii))  
}


####
## Apply ADMM 
####
