## check some kronecker thing

library(pracma)
library(magrittr)
library(plyr)

n <- 10
v <- 2
rk <- 2

U <- randn(n, rk)
V <- randn(rk, v)
E <- U %*% V
Evec <- E %>% as.numeric

Ukron <- eye(v) %x% U
Vvec <- as.numeric(V)
Evec1 <- Ukron %*% Vvec
Norm(Evec - Evec1)

Vkron <- do.call(rbind, alply(V, 2, function(x) eye(n) %x% t(x)))
Uvec <- t(U) %>% as.numeric
Evec2 <- Vkron %*% Uvec
Norm(Evec - Evec2)

Vkron <- t(V) %x% eye(n)
Uvec <- U %>% as.numeric
Evec2 <- Vkron %*% Uvec
Norm(Evec - Evec2)
