tab <- read.table('~/predator//clusters.csv', header = FALSE, sep = " ")
table(tab[, 3])
tab[, 5] <- round(tab[, 5])
nms <- list("cluster", c("x", "y", "z"), "wm", paste0("A", 1:150), paste0("B", 1:150),
            paste0("C", 1:46), paste0("D", 1:46))
colnames(tab) <- do.call(c, nms)
tab[, 1] <- tab[, 1] + 1
head(tab)
saveRDS(tab, "data/clusters.rds")
