####
## Nuclear norm analysis
####

source('theory/dwi.R')
bv1 <- readRDS('data/chris_bv1.rds')
bv2 <- readRDS('data/chris_bv2.rds')
set.seed(2)
pts <- metasub(xyz, 0.009, 10)
dim(pts)

clusters <- readRDS('data/clusters.rds')
table(clusters[, 1])
cll <- lapply(1:54, function(i) clusters[clusters[,1] == i, ])
cll[[1]][1:10, 1:10]
cll[[2]][1:10, 1:10]
