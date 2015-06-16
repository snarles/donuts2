####
## Nuclear norm analysis
####

library(magrittr)

source('theory/dwi.R')
bv1 <- readRDS('data/chris_bv1.rds')
bv2 <- readRDS('data/chris_bv2.rds')
set.seed(2)
pts <- metasub(xyz, 0.009, 10)
dim(pts)

clusters <- readRDS('data/clusters.rds')
table(clusters[, 1])
cll <- lapply(1:47, function(i) clusters[clusters[,1] == i, ])

###
## Setup
###

kappa <- 3
dA <- paste0("A", which(rownames(bv1)=="d"))
dB <- paste0("B", which(rownames(bv1)=="d"))
dv1 <- bv1[rownames(bv1) == "d", ]
dv2 <- bv2[rownames(bv1) == "d", ]
Xa <- stetan(dv1, pts, kappa)
Xb <- stetan(dv1, pts, kappa)
mc.cores = 7

###
## Analyze one cluster
###

ind <- 1
cl <- cll[[ind]]
wms = cl[, "wm"]
Ya <- t(cl[, dA])
Yb <- t(cl[, dB])
nv <- dim(Ya)[2]

Ba_nnls <- multi_nnls(Xa, Ya, mc.cores = mc.cores)
YhB_nnls <- Xb %*% Ba_nnls
Bb_nnls <- multi_nnls(Xb, Yb, mc.cores = mc.cores)
YhA_nnls <- Xa %*% Bb_nnls

(objA_nnls <- sum((Xa %*% Ba_nnls - Ya)^2))
(objB_nnls <- sum((Xb %*% Bb_nnls - Yb)^2))
(errA_nnls <- sum((Ya - YhA_nnls)^2))
(errB_nnls <- sum((Yb - YhB_nnls)^2))
(errAw_nnls <- sum((Ya - YhA_nnls)[, wms == 1]^2))
(errBw_nnls <- sum((Yb - YhB_nnls)[, wms == 1]^2))


# constrant for NN
lambda <- 10
resA_nn <- nuclear_opt(Xa, Ya, lambda, mc.cores = mc.cores)
resB_nn <- nuclear_opt(Xb, Yb, lambda, mc.cores = mc.cores)

Ba_nn <- resA_nn$B
YhB_nn <- Xb %*% Ba_nn
Bb_nn <- resB_nn$B
YhA_nn <- Xa %*% Bb_nn

(errA_nn <- sum((Ya - YhA_nn)^2))
(errB_nn <- sum((Yb - YhB_nn)^2))
(errAw_nn <- sum((Ya - YhA_nn)[, wms == 1]^2))
(errBw_nn <- sum((Yb - YhB_nn)[, wms == 1]^2))


