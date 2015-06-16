
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

kappa <- 4
l1p <- 2
dA <- paste0("A", which(rownames(bv1)=="d"))
dB <- paste0("B", which(rownames(bv1)=="d"))
dv1 <- bv1[rownames(bv1) == "d", ]
dv2 <- bv2[rownames(bv1) == "d", ]
Xa <- rbind(cbind(1, stetan(dv1, pts, kappa)), c(0, rep(l1p, 1000)))
Xb <- rbind(cbind(1, stetan(dv1, pts, kappa)), c(0, rep(l1p, 1000)))
mc.cores = 14

sapply(cll, function(v) sum(v[, "wm"]))

###
## Analyze one cluster
###

ind <- 6
cl <- cll[[ind]]
wms = cl[, "wm"]
Ya <- rbind(t(cl[, dA]), 0)
Yb <- rbind(t(cl[, dB]), 0)
nv <- dim(Ya)[2]
Ba_nnls <- multi_nnls(Xa, Ya, mc.cores = mc.cores)
YhB_nnls <- Xb %*% Ba_nnls
Bb_nnls <- multi_nnls(Xb, Yb, mc.cores = mc.cores)
YhA_nnls <- Xa %*% Bb_nnls
(objA_nnls <- sum((Xa %*% Ba_nnls - Ya)^2))
(objB_nnls <- sum((Xb %*% Bb_nnls - Yb)^2))
#(errA_nnls <- sum((Ya - YhA_nnls)^2))
#(errB_nnls <- sum((Yb - YhB_nnls)^2))
(errAw_nnls <- sum((Ya - YhA_nnls)[1:138, wms == 1]^2))
(errBw_nnls <- sum((Yb - YhB_nnls)[1:138, wms == 1]^2))
# constrant for NN
lambda <- 3
t1 <- proc.time()[3]
resA_nn <- nuclear_opt(Xa, Ya, lambda, 30, mc.cores = mc.cores)
resB_nn <- nuclear_opt(Xb, Yb, lambda, 30, mc.cores = mc.cores)
proc.time()[3] - t1
Ba_nn <- resA_nn$B
YhB_nn <- Xb %*% Ba_nn
Bb_nn <- resB_nn$B
YhA_nn <- Xa %*% Bb_nn
#(errA_nn <- sum((Ya - YhA_nn)^2))
#(errB_nn <- sum((Yb - YhB_nn)^2))
(errAw_nn <- sum((Ya - YhA_nn)[1:138, wms == 1]^2))
(errBw_nn <- sum((Yb - YhB_nn)[1:138, wms == 1]^2))
(floor(svd(resA_nn$E)$d * 100)/100)[1:10]
#resSize <- multi_nnls(Xa, resA_nn$E)
#sum(abs(resSize))
sum(abs(resA_nn$B))
sum(abs(Ba_nnls))

