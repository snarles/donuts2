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

# the expectation of the absolute value of a normal
sqrt(2/pi)
mean(abs(rnorm(100000)))

# generates n columns of p-vectors x_i, based on
# x0 = abs(rnorm(p)), x=x/sum(x0)
rand.vec <- function(n,p) {
  res <- matrix(0,p,n)
  for (i in 1:n) {
    x0 <- abs(rnorm(p))
    x <- x0/sum(x0)
    res[,i] <- x
  }
  return(res)
}

# claim: as p to infinity, the distance of x to (1/p,...,1/p) has mean sqrt((pi/2-1)/p) and variance O(1/p^2)

p <- 2000
n <- 1000
res <- rand.vec(n,p)
dists <- sqrt(apply((res - 1/p)^2,2,sum))
mean(dists)
sqrt((pi/2 - 1)/p)
var(dists)*p^2 # around 0.3

max(dists)-mean(dists)
min(dists)-mean(dists)

# the above in function form
gen.dists <- function(n,p) {
  res <- rand.vec(n,p)
  dists <- sqrt(apply((res - 1/p)^2,2,sum))
  return(dists)
}

gen.dists(10,100)

# claim: with probability approaching 1 given the right rate of growth for n, p,
# the maximimum distance of {x_1,..,x_n} to (1/p,...,1/p) is bounded by the mean distance plus O(sqrt(2log(n))/p)
# and a similar statement holds for minimum distance

n <- 100
p <- 1000
n.its <- 100
maxs <- numeric(n.its)
mins <- numeric(n.its)
meanz <- numeric(n.its)
for (j in 1:n.its) {
  dists <- gen.dists(n,p)
  maxs[j] <- max(dists)
  mins[j] <- min(dists)
  meanz[j] <- mean(dists)
}
emp.mean <- mean(meanz)
sort(maxs-emp.mean)/(sqrt(2*log(n))/p) # between 0.3 and 0.8
sort(mins-emp.mean)/(sqrt(2*log(n))/p) # between -0.8 and -0.3
