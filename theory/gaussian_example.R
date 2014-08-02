# generates supremum of p standard gaussian variates
generate.gsup <- function(n, p) {
  res <- numeric(n)
  for (i in 1:n) {
    res[i] <- max(rnorm(p))
  }
  return(res)
}

mean(generate.gsup(1000,100))
var(generate.gsup(1000,100))

# generates n columns of p-vectors x_i, based on
# x0 = abs(rnorm(p)), x=x/sum(x0)
rand.vec <- function(n,p) {
  res <- matrix(0,n,p)
  for (i in 1:n) {
    x0 <- abs(rnorm(p))
    x <- x0/sum(x0)
    res[,i] <- x
  }
}
