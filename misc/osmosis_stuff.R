####
## Reading diffusion data from HCP
####

library(oro.nifti)
library(magrittr)
library(rgl)
library(AnalyzeFMRI)
library(pracma)
source("theory/dwi.R")

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

#ddir <- function(s = "") paste0("/home/snarles/hcp/10s/115320/", s)
ddir <- function(s = "") paste0("/home/snarles/predator/osmosis/", s)


#list.files(ddir("T1w/Diffusion"))
list.files(ddir())


## ** B-values **

set.seed(2)
pts <- metasub(xyz, 0.009, 10)
p <- dim(pts)[1]
plot3d(pts)

#bvals <- read.table(ddir("T1w/Diffusion/bvals"), header = FALSE, sep = "") %>% as.numeric
bvi <- round(bvals/1000)
#bvecs <- read.table(ddir("T1w/Diffusion/bvecs"), header = FALSE, sep = "") %>% 
bvecs1 <- (read.table(ddir("SUB1_b1000_1.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix)[11:160, ]
bvecs2 <- (read.table(ddir("SUB1_b1000_2.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix)[11:160, ]

plot3d(bvecs1); points3d(1.01 * bvecs2, col = "red")

## ** nifti **



(wm1 <- readNIfTI(ddir("SUB1_wm_mask.nii.gz")))
(diff1 <- readNIfTI(ddir("SUB1_b1000_1.nii.gz")))
(wm2 <- readNIfTI(ddir("SUB2_wm_mask.nii.gz")))


dim(wm1) # 81 106 76
dim(wm2) # 81 106 76

gimage(wm1[, , 30])
gimage(wm2[, , 30])

gimage(wm1[, , 40])
gimage(wm2[, , 40])

## wm consensus
wmc <- wm1 * wm2

gimage(wmc[, , 40])

## smoothed WM (to find wm ROIs)

wms <- GaussSmoothArray(wm1 + wm2, ksize = 17)
dim(wms)

max(wms)
gimage(wms[, , 40])
sum(wms > 1.99)

roi_inds <- which(wms > 1.99, arr.ind = TRUE)
plot3d(roi_inds, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76))

temp <- extract_vox(diff1, roi_inds)
diff1r <- temp[, 11:160]
so1r <- temp[, 1:10]

kappa <- 1
X1 <- cbind(1, stetan(bvecs1, pts, kappa))
Y <- t(diff1r)#[, 1:10]
B <- multi_nnls(X1, Y, mc.cores = 3)
mu <- X1 %*% B
resid <- Y - mu

dim(resid)
res <- svd(resid)
dim(res$u)
plot(res$u[, 1])

cols <- hsv(h = 0.5, s = 1, v = 1:150/150)

plot3d(bvecs1, col = cols[order(res$u[, 1])], size = 10)
plot3d(bvecs1, col = cols[order(res$u[, 2])], size = 10)
