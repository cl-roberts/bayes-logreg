#------------------------------------------------------------------------------#

# Bayesian logistic regression example in TMB

# r file

# CL Roberts

# 6/11/2025

#------------------------------------------------------------------------------#

## MCMC controls ---

chains <- 1
set.seed(406)
seeds <- sample(1:1e4, size = chains)
iter <- 1000
warmup <- 500
control <- list()
# control <- list(adapt_delta = 0.95)


## set up ---

# pkgs
library(TMB)
library(tmbstan)
library(here)

# dir 

dir_ich <- here()
dir_src <- here("src")

## simulate data ---

set.seed(406)
n <- 100

# regressors

x <- rnorm(n)
beta0 <- 1
beta1 <- 2

# response

logit_p <- beta0 + beta1*x + rnorm(n, sd = .1)
p <- 1/(1+exp(-logit_p))

size <- rep(10, n)
y <- rbinom(n = n, size = size, p = p)

# priors

beta1_prior <- c(10, 1) # dnorm - mean, sd
beta2_prior <- c(10, 1)

da <- list(y = y, x = x, size = size, 
           beta1_prior = beta1_prior,
           beta2_prior = beta2_prior)

## Compile and load model ---
if("logreg" %in% names(getLoadedDLLs())) {
    dyn.unload(dynlib(here(dir_src, "logreg")))
}
compile(here(dir_src, "logreg.cpp"))
dyn.load(dynlib(here(dir_src, "logreg")))

## initial parameter values ---
inits <- list(beta = c(2, 2))

## create model object ---

obj <- MakeADFun(
    data = da,
    parameters = inits,
    DLL = "logreg",
    hessian = TRUE
)

## save initial fit ---

before.optim <- c(
    last.par = obj$env$last.par,
    par = obj$par,
    fn = obj$fn(),
    gr = obj$gr()
)

## ML optimization ---
opt <- nlminb(obj$par, obj$fn, obj$gr)
opt$par
sdreport(obj)

## run NUTS ---
mcmc_start_time <- Sys.time()
fit <- tmbstan(obj, chains = chains, cores = chains, init = inits, iter = iter, 
                warmup = warmup, seed = seeds, algorithm = "NUTS", silent = FALSE,
                control = control)
mcmc_end_time <- Sys.time()
time <- mcmc_end_time - mcmc_start_time

# save parameter posteriors ---
post <- as.data.frame(fit)
mcmc_results <- as.data.frame(fit)[,-ncol(post)]
apply(mcmc_results, MARGIN = 2, FUN = median)
apply(mcmc_results, MARGIN = 2, FUN = sd)

# plot parameters ----

hist(mcmc_results[,1])
traceplot(fit, pars=names(obj$par), inc_warmup=TRUE)


# plot fit ---

plot_data <- data.frame(x = x, y = y)

yhat_post <- apply(
        mcmc_results,
        MARGIN = 1,
        FUN = \(x) obj$report(x)$yhat
    ) |>
    t()

plot_data$yhat <- apply(yhat_post, MARGIN = 2, FUN = median)
plot_data$yhat_lwr <- apply(yhat_post, MARGIN = 2, FUN = \(x) quantile(x, .025))
plot_data$yhat_upr <- apply(yhat_post, MARGIN = 2, FUN = \(x) quantile(x, .975))

plot_data$y_true <- obj$report(c(beta0, beta1))$yhat

plot_data <- plot_data[order(plot_data$x),]

plot(plot_data$x, plot_data$y)
lines(plot_data$x, plot_data$yhat, type = "l")
lines(plot_data$x, plot_data$y_true, type = "l", col = "red")
lines(plot_data$x, plot_data$yhat_lwr, type = "l", lty = 2)
lines(plot_data$x, plot_data$yhat_upr, type = "l", lty = 2)


# test for overdispersion ---

phat_post <- apply(
        mcmc_results,
        MARGIN = 1,
        FUN = \(x) obj$report(x)$phat
    ) |>
    t()

phat <- apply(phat_post, MARGIN = 2, FUN = median)

chi_sq <- sum((y - size*phat)^2 / (size*phat*(1-phat)))

pchisq(chi_sq, n - ncol(mcmc_results), lower.tail = FALSE)
