x = 0:99
require(stats)
require(splines)
res = bs(x,df=10)
dim(res)
plot(x,res[,3])



# source code

function (x, df = NULL, knots = NULL, degree = 3, intercept = FALSE, 
    Boundary.knots = range(x)) 
{
    nx <- names(x)
    x <- as.vector(x)
    nax <- is.na(x)
    if (nas <- any(nax)) 
        x <- x[!nax]
    if (!missing(Boundary.knots)) {
        Boundary.knots <- sort(Boundary.knots)
        outside <- (ol <- x < Boundary.knots[1L]) | (or <- x > 
            Boundary.knots[2L])
    }
    else outside <- FALSE
    ord <- 1L + (degree <- as.integer(degree))
    if (ord <= 1) 
        stop("'degree' must be integer >= 1")
    if (!is.null(df) && is.null(knots)) {
        nIknots <- df - ord + (1L - intercept)
        if (nIknots < 0L) {
            nIknots <- 0L
            warning(gettextf("'df' was too small; have used %d", 
                ord - (1L - intercept)), domain = NA)
        }
        knots <- if (nIknots > 0L) {
            knots <- seq.int(from = 0, to = 1, length.out = nIknots + 
                2L)[-c(1L, nIknots + 2L)]
            stats::quantile(x[!outside], knots)
        }
    }
    Aknots <- sort(c(rep(Boundary.knots, ord), knots))
    if (any(outside)) {
        warning("some 'x' values beyond boundary knots may cause ill-conditioned bases")
        derivs <- 0:degree
        scalef <- gamma(1L:ord)
        basis <- array(0, c(length(x), length(Aknots) - degree - 
            1L))
        if (any(ol)) {
            k.pivot <- Boundary.knots[1L]
            xl <- cbind(1, outer(x[ol] - k.pivot, 1L:degree, 
                "^"))
            tt <- splineDesign(Aknots, rep(k.pivot, ord), ord, 
                derivs)
            basis[ol, ] <- xl %*% (tt/scalef)
        }
        if (any(or)) {
            k.pivot <- Boundary.knots[2L]
            xr <- cbind(1, outer(x[or] - k.pivot, 1L:degree, 
                "^"))
            tt <- splineDesign(Aknots, rep(k.pivot, ord), ord, 
                derivs)
            basis[or, ] <- xr %*% (tt/scalef)
        }
        if (any(inside <- !outside)) 
            basis[inside, ] <- splineDesign(Aknots, x[inside], 
                ord)
    }
    else basis <- splineDesign(Aknots, x, ord)
    if (!intercept) 
        basis <- basis[, -1L, drop = FALSE]
    n.col <- ncol(basis)
    if (nas) {
        nmat <- matrix(NA, length(nax), n.col)
        nmat[!nax, ] <- basis
        basis <- nmat
    }
    dimnames(basis) <- list(nx, 1L:n.col)
    a <- list(degree = degree, knots = if (is.null(knots)) numeric(0L) else knots, 
        Boundary.knots = Boundary.knots, intercept = intercept)
    attributes(basis) <- c(attributes(basis), a)
    class(basis) <- c("bs", "basis", "matrix")
    basis
}


function (knots, x, ord = 4, derivs = integer(nx), outer.ok = FALSE, 
    sparse = FALSE) 
{
    knots <- sort(as.numeric(knots))
    if ((nk <- length(knots)) <= 0) 
        stop("must have at least 'ord' knots")
    x <- as.numeric(x)
    nx <- length(x)
    if (length(derivs) != nx) 
        stop("length of 'derivs' must match length of 'x'")
    ord <- as.integer(ord)
    if (ord > nk || ord < 1) 
        stop("'ord' must be positive integer, at most the number of knots")
    if (!outer.ok && nk < 2 * ord - 1) 
        stop(gettextf("need at least %s (=%d) knots", "2*ord -1", 
            2 * ord - 1), domain = NA)
    o1 <- ord - 1L
    if (need.outer <- any(out.x <- x < knots[ord] | knots[nk - 
        o1] < x)) {
        if (outer.ok) {
            in.x <- knots[1L] < x & x < knots[nk]
            knots <- knots[c(rep.int(1L, o1), 1L:nk, rep.int(nk, 
                o1))]
            if ((x.out <- !all(in.x))) {
                x <- x[in.x]
                nnx <- length(x)
            }
        }
        else stop(gettextf("the 'x' data must be in the range %g to %g unless you set '%s'", 
            knots[ord], knots[nk - o1], "outer.ok = TRUE"), domain = NA)
    }
    temp <- .Call(C_spline_basis, knots, ord, x, derivs)
    ncoef <- nk - ord
    ii <- if (need.outer && x.out) {
        rep.int((1L:nx)[in.x], rep.int(ord, nnx))
    }
    else rep.int(1L:nx, rep.int(ord, nx))
    jj <- c(outer(1L:ord, attr(temp, "Offsets"), "+"))
    if (sparse) {
        if (is.null(tryCatch(loadNamespace("Matrix"), error = function(e) NULL))) 
            stop(gettextf("%s needs package 'Matrix' correctly installed", 
                "splineDesign(*, sparse=TRUE)"), domain = NA)
        if (need.outer) {
            jj <- jj - o1 - 1L
            ok <- 0 <= jj & jj < ncoef
            as(new("dgTMatrix", i = ii[ok] - 1L, j = jj[ok], 
                x = as.double(temp[ok]), Dim = c(nx, ncoef)), 
                "CsparseMatrix")
        }
        else as(new("dgTMatrix", i = ii - 1L, j = jj - 1L, x = as.double(temp), 
            Dim = c(nx, ncoef)), "CsparseMatrix")
    }
    else {
        design <- matrix(double(nx * ncoef), nx, ncoef)
        if (need.outer) {
            jj <- jj - o1
            ok <- 1 <= jj & jj <= ncoef
            design[cbind(ii, jj)[ok, , drop = FALSE]] <- temp[ok]
        }
        else design[cbind(ii, jj)] <- temp
        design
    }
}
<bytecode: 0x12993d8>
<environment: namespace:splines>
