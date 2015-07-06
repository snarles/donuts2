####
## Reading diffusion data from HCP
####

library(oro.nifti)
library(magrittr)
library(rgl)
library(AnalyzeFMRI)
library(pracma)

gimage <- function(a) image(fliplr(t(a)), col = gray(0:20/20))

#ddir <- function(s = "") paste0("/home/snarles/hcp/10s/115320/", s)
ddir <- function(s = "") paste0("/home/snarles/predator/osmosis/", s)


#list.files(ddir("T1w/Diffusion"))
list.files(ddir())


## ** B-values **

#bvals <- read.table(ddir("T1w/Diffusion/bvals"), header = FALSE, sep = "") %>% as.numeric
bvi <- round(bvals/1000)
#bvecs <- read.table(ddir("T1w/Diffusion/bvecs"), header = FALSE, sep = "") %>% 
bvecs1 <- read.table(ddir("SUB1_b1000_1.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix
bvecs2 <- read.table(ddir("SUB1_b1000_2.bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix

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
sum(wms > 1.9)

roi_inds <- which(wms > 1.9, arr.ind = TRUE)
plot3d(roi_inds, xlim = c(1, 81), ylim = c(1, 106), zlim = c(1, 76))

help(row)
