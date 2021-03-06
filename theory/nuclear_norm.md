---
title: "NNLS with Nuclear Norm"
author: "Charles Zheng"
date: "06/13/2015"
output: html_document
---

Consider the problem

$$
Y = XB + E
$$
where $Y_{n \times v}$, $X_{n \times p}$ are known, while $B_{p \times v} \geq 0$ and $E_{n \times v}$ low-rank are unknown.

Our goal is to recover the unknown $B$ from observed $Y$ and $X$.
We solve the convex relaxation
$$
\text{minimize } ||Y - XB - E||_F^2 \text{ subject to }B\geq 0,\ ||E||_* \leq c
$$
where $||\cdot||_*$ is the nuclear norm.

Alternatively, define $P_C$ to be the projection onto the cone $C = \{V: V = XU\text{ for }U \geq 0\}$.
Then one way of solving the optimization problem is to define $V = Y-E$ and solve
$$
\text{minimize } ||V - P_C V||^2_F \text{ subject to }||Y - V||_* \leq c
$$
then finding $B$ which minimizes $||V - XB||^2_F$ subject to $B \geq 0$.

## General Facts

The previous problem takes the form
$$
\text{minimize } ||v  - P_C v||^2 \text{ subject to }v \in S
$$
where $C$ and $S$ are convex sets.  Assume for now that $C$ is smooth, and that the interior of $C$ is nonempty.

We can evaluate the gradient as follows.

The property of the projection $P_C v$ is that $v - P_C v \perp P_C v - c$ for all $c \in C$.
This implies that
$$
\frac{dP_c v}{dv} = (I-P_\delta)
$$
where $P_\delta$ is projection onto the line spanned by $v - P_C v$.

$$
\frac{d}{dv} ||v - P_C v||^2 = \frac{d}{dv} (v'v - 2v'P_C v + (P_C v)'(P_C v))
$$
$$
= 2(v - P_C v ) - 2(I- P_\delta)v + 2(I - P_\delta)P_C v
$$
$$
= 2(v - P_C v ) - 2(I- P_\delta)(v - P_C v) = 2(v - P_C v )
$$
since $(I - P_\delta) (v - P_C v) = 0$.

## Algorithm

Let us return to the specific problem
$$
\text{minimize } ||V - P_C V||^2_F \text{ subject to }||Y - V||_* \leq c
$$

As we can see, the gradient of the unconstrained problem is simply a multiple of $V - P_C V$.
Meanwhile, we have the following algorithm due to Jaggi (2010) for minimizing convex objectives subject to nuclear norm constraint $||V||_* \leq c$

* Initialize $V = 0$
* For iteration $k = 1, ... $:
* Let $\alpha = 1/k$
* Evaulate the unconstrained gradient $G$
* Let $H$ be the rank-1 approximation of $-G$, scaled so that $||H||_* = c$
* Update $V = (1-\alpha) V + \alpha H$

## Example Problem

Requirements


```r
library(pracma)
library(magrittr)
library(nnls)
library(parallel)
```

Let X be randomly generated, B sparse, E rank 2.


```r
n <- 30
p <- 20
v <- 10
sparsity <- 0.1
sigma <- 0.1
rk <- 3
X <- randn(n, p) %>% abs
B <- randn(p, v) %>% abs * (rand(p, v) < sparsity)
U <- sqrt(sigma/rk) * randn(n, rk)
V <- sqrt(sigma/rk) * randn(rk, v)
E <- U %*% V
mu <- X %*% B
Y <- mu + E
```

### Use NNLS to recover B

```r
multi_nnls <- function(X, Y, mc.cores = 3) {
  v <- dim(Y)[2]
  bs <- mclapply(1:v, function(i) nnls(X, Y[, i])$x, mc.cores = mc.cores)
  do.call(cbind, bs)
}
B_nnls <- multi_nnls(X, Y)
#Check denoising error
mu_nnls <- X %*% B_nnls
(err_nnls <- sum((mu_nnls - mu)^2))
```

```
## [1] 0.1316546
```

### Use biconvex optimization to recover B


```r
biconvex_opt <- function(X, Y, rk = 2, maxits = 10) {
  # initializiation
  n <- dim(Y)[1]
  p <- dim(X)[2]
  v <- dim(Y)[2]
  U <- randn(n, rk)
  V <- randn(rk, v)
  B <- randn(p, v) %>% abs
  res <- list(Bvec = as.numeric(B), 
              Uvec = as.numeric(U),
              Vvec = as.numeric(V))
  Yvec <- as.numeric(Y)
  Xkron <- diag(rep(1, v)) %x% X
  # subroutines
  objective <- function(Bvec, Uvec, Vvec) {
    resid <- matrix(Y, n, v) - X %*% matrix(Bvec, p, v) -
      matrix(Uvec, n, rk) %*% matrix(Vvec, rk, v)
    sum(resid^2)
  }
  biconvex_it1 <- function(Bvec, Uvec, Vvec) {
    U <- matrix(Uvec, n, rk)
    Ukron <- eye(v) %x% U
    supermat <- cbind(Xkron, Ukron, -Ukron)
    ans <- nnls(supermat, Yvec)$x
    Bvec <- ans[1:(p * v)]
    VvecP <- ans[p * v + (1:(v * rk))]
    VvecN <- ans[p * v + v * rk + (1:(v * rk))]
    Vvec <- VvecP - VvecN
    list(Bvec = Bvec, Uvec = Uvec, Vvec = Vvec)
  }
  biconvex_it2 <- function(Bvec, Uvec, Vvec) {
    V <- matrix(Vvec, rk, v)
    Vkron <- t(V) %x% eye(n)
    supermat <- cbind(Xkron, Vkron, -Vkron)
    ans <- nnls(supermat, Yvec)$x
    Bvec <- ans[1:(p * v)]
    UvecP <- ans[p * v + (1:(n * rk))]
    UvecN <- ans[p * v + n * rk + (1:(n * rk))]
    Uvec <- UvecP - UvecN
    list(Bvec = Bvec, Uvec = Uvec, Vvec = Vvec)
  }
  # main loop
  objs <- Inf
  for (i in 1:maxits) {
    (objs <- objs %>% c(do.call(objective, res)))
    res <- do.call(biconvex_it1, res)
    (objs <- objs %>% c(do.call(objective, res)))
    res <- do.call(biconvex_it2, res)
    (objs <- objs %>% c(do.call(objective, res)))
  }
  B <- matrix(res$Bvec, p, v)
  E <- matrix(res$Uvec, n, rk) %*% matrix(res$Vvec, rk, v)
  list(B = B, E = E, mu = X %*% B, resid = Y - X %*% B - E, objs = objs,
       U = matrix(res$Uvec, n, rk), V = matrix(res$Vvec, rk, v),
       objective = objective)
}
res_bc <- biconvex_opt(X, Y, 1, 10)
# compare true size of E with fitted size
c(sum(E^2), sum(res_bc$E^2))
```

```
## [1] 0.988448 0.779417
```

```r
B_bc <- res_bc$B
# compare the objective of the truth with the objective
c(res_bc$objective(as.numeric(B), as.numeric(U), as.numeric(V)),
  min(res_bc$objs))
```

```
## [1] 1.199155 0.240764
```

```r
#Check denoising error
mu_bc <- X %*% B_bc
(err_bc <- sum((mu_bc - mu)^2))
```

```
## [1] 0.05791189
```

### Use regularized biconvex optimization to recover B

Put an L2 penalty on $U$ and $V$ and also an L1 penalty on $B$.


```r
reg_biconvex_opt <- function(X, Y, rk = 2, maxits = 10,
                             lambdaU = 0.1, lambdaV = 0.1,
                             lambdaB = 0.1) {
  # initializiation
  n <- dim(Y)[1]
  p <- dim(X)[2]
  v <- dim(Y)[2]
  U <- randn(n, rk)
  V <- randn(rk, v)
  B <- randn(p, v) %>% abs
  res <- list(Bvec = as.numeric(B), 
              Uvec = as.numeric(U),
              Vvec = as.numeric(V))
  Yvec <- as.numeric(Y)
  Xkron <- diag(rep(1, v)) %x% X
  # subroutines
  objective <- function(Bvec, Uvec, Vvec) {
    resid <- matrix(Y, n, v) - X %*% matrix(Bvec, p, v) -
      matrix(Uvec, n, rk) %*% matrix(Vvec, rk, v)
    sum(resid^2) + lambdaU * sum(Uvec^2) + lambdaV * sum(Vvec^2) + 
      lambdaB * sum(Bvec)^2
  }
  biconvex_it1 <- function(Bvec, Uvec, Vvec) {
    U <- matrix(Uvec, n, rk)
    Yvec2 <- c(Yvec, rep(0, 1 + 2 * length(Vvec)))
    Ukron <- eye(v) %x% U
    supermat <- rbind(cbind(Xkron, Ukron, -Ukron),
                      c(rep(sqrt(lambdaB), length(Bvec)), rep(0, 2 * length(Vvec))),
                      cbind(matrix(0, 2 * length(Vvec), dim(Xkron)[2]),
                            sqrt(lambdaV) * eye(2 * length(Vvec))))
    ans <- nnls(supermat, Yvec2)$x
    Bvec <- ans[1:(p * v)]
    VvecP <- ans[p * v + (1:(v * rk))]
    VvecN <- ans[p * v + v * rk + (1:(v * rk))]
    Vvec <- VvecP - VvecN
    list(Bvec = Bvec, Uvec = Uvec, Vvec = Vvec)
  }
  biconvex_it2 <- function(Bvec, Uvec, Vvec) {
    V <- matrix(Vvec, rk, v)
    Yvec2 <- c(Yvec, rep(0, 1 + 2 * length(Uvec)))
    Vkron <- t(V) %x% eye(n)
    supermat <- rbind(cbind(Xkron, Vkron, -Vkron),
                      c(rep(sqrt(lambdaB), length(Bvec)), rep(0, 2 * length(Uvec))),
                      cbind(
                        matrix(0, 2 * length(Uvec), dim(Xkron)[2]),
                        sqrt(lambdaU) * eye(2 * length(Uvec))
                      ))
    ans <- nnls(supermat, Yvec2)$x
    Bvec <- ans[1:(p * v)]
    UvecP <- ans[p * v + (1:(n * rk))]
    UvecN <- ans[p * v + n * rk + (1:(n * rk))]
    Uvec <- UvecP - UvecN
    list(Bvec = Bvec, Uvec = Uvec, Vvec = Vvec)
  }
  # main loop
  objs <- Inf
  for (i in 1:maxits) {
    (objs <- objs %>% c(do.call(objective, res)))
    res <- do.call(biconvex_it1, res)
    (objs <- objs %>% c(do.call(objective, res)))
    res <- do.call(biconvex_it2, res)
    (objs <- objs %>% c(do.call(objective, res)))
  }
  B <- matrix(res$Bvec, p, v)
  E <- matrix(res$Uvec, n, rk) %*% matrix(res$Vvec, rk, v)
  list(B = B, E = E, mu = X %*% B, resid = Y - X %*% B - E, objs = objs,
       U = matrix(res$Uvec, n, rk), V = matrix(res$Vvec, rk, v),
       objective = objective)
}
res_bc2 <- reg_biconvex_opt(X, Y, 2, 10, lambdaU = 0.001, lambdaV = 0.001,
                            lambdaB = 0.0)
# compare true size of E with fitted size
c(sum(E^2), sum(res_bc2$E^2))
```

```
## [1] 0.9884480 0.9069985
```

```r
B_bc2 <- res_bc2$B
# compare the objective of the truth with the objective
c(res_bc2$objective(as.numeric(B), as.numeric(U), as.numeric(V)),
  min(res_bc2$objs))
```

```
## [1] 1.60595989 0.09673082
```

```r
#Check denoising error
mu_bc2 <- X %*% B_bc2
(err_bc2 <- sum((mu_bc2 - mu)^2))
```

```
## [1] 0.08728412
```

Better...?


```r
err_nnls
```

```
## [1] 0.1316546
```

```r
err_bc
```

```
## [1] 0.05791189
```

```r
err_bc2
```

```
## [1] 0.08728412
```

### Nuclear norm optimization


```r
nuclear_opt <- function(X, Y, lambda, mc.cores = 3) {
  
}
```
