####
## Reading diffusion data from HCP
####

library(oro.nifti)
library(magrittr)
library(rgl)

ddir <- function(s = "") paste0("/home/snarles/hcp/10s/115320/", s)

list.files(ddir("T1w/Diffusion"))

## ** B-values **

bvals <- read.table(ddir("T1w/Diffusion/bvals"), header = FALSE, sep = "") %>% as.numeric
bvi <- round(bvals/1000)
bvecs <- read.table(ddir("T1w/Diffusion/bvecs"), header = FALSE, sep = "") %>% 
  t %>% as.matrix
plot3d(bvecs[bvi == 1, ])
points3d(bvecs[bvi == 2, ], col = "red")
points3d(bvecs[bvi == 3, ], col = "blue")
comb <- cbind(bvi, bvecs)

## ** nifti **

(hcp1 <- readNIfTI(ddir("T1w/Diffusion/data.nii.gz")))
