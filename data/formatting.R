library(magrittr)

tab <- read.table('~/predator//clusters.csv', header = FALSE, sep = " ")
tab %<>% as.matrix
table(tab[, 3])
tab[, 5] <- round(tab[, 5])
nms <- list("cluster", c("x", "y", "z"), "wm", paste0("A", 1:150), paste0("B", 1:150),
            paste0("C", 1:46), paste0("D", 1:46))
colnames(tab) <- do.call(c, nms)
tab[, 1] <- tab[, 1] + 1
head(tab)
saveRDS(tab, "data/clusters.rds")

bv1 <- read.table("data//chris1_bvec.csv", header = FALSE, sep = " ")
bv1 %<>% t
nms <- rep("d", 150)
nms[rowSums(bv1) == 0] <- "b0"
nms[1:2] <- "b0cal"
rownames(bv1) <- nms
saveRDS(bv1, "data/chris_bv1.rds")

bv2 <- read.table("data//chris2_bvec.csv", header = FALSE, sep = " ")
bv2 %<>% t
nms <- rep("d", 150)
nms[rowSums(bv2) == 0] <- "b0"
nms[1:2] <- "b0cal"
rownames(bv2) <- nms
saveRDS(bv2, "data/chris_bv2.rds")
