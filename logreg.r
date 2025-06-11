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

## set up ---

# pkgs
library(TMB)
library(tmbstan)
library(here)
library(ggplot2)

# dir 

dir_ich <- here()
dir_src <- here("src")

## simulate data ---

set.seed(406)
n <- 50              # number samples per random effect
num_groups <- 10      # number of groups in random effect

# regressors

x <- cbind(1, rnorm(n*num_groups), rep(1:num_groups, rep(n, num_groups)))

# parameters 
mu <- c(0, 5)         # mean of random intercept and slope
sigma <- c(1, 3)     # variance of random intercept and slope

beta <- cbind(
    rnorm(num_groups, mean = mu[1], sd = sigma[1]), 
    rnorm(num_groups, mean = mu[2], sd = sigma[2])
)

# response

logit_p <- c()
for (group in 1:num_groups) {
    logit_p <- c(
        logit_p, 
        x[x[,3] == group,1:2] %*% beta[group,] + rnorm(n, sd = 1)
    )
}

p <- 1/(1+exp(-logit_p))

size <- rep(10, n*num_groups)
y <- rbinom(n = n*num_groups, size = size, p = p)

## data ----

da <- list(y = y, x = x[,2], size = size, group = x[,3])

# priors

beta0_prior <- c(0, 3) # dnorm - mean, sd
beta1_prior <- c(0, 3)
mu0_prior <- c(0, 3) 
sigma0_prior <- c(1, 3) 
mu1_prior <- c(5, 3) 
sigma1_prior <- c(3, 3) 

priors <- list(
    beta0_prior = beta0_prior, beta1_prior = beta1_prior, 
    mu0_prior = mu0_prior, sigma0_prior = sigma0_prior, 
    mu1_prior = mu1_prior, sigma1_prior = sigma1_prior
)

# all inputs

input <- c(da, priors)

## Compile and load model ---
if("logreg" %in% names(getLoadedDLLs())) {
    dyn.unload(dynlib(here(dir_src, "logreg")))
}
compile(here(dir_src, "logreg.cpp"))
dyn.load(dynlib(here(dir_src, "logreg")))

## parameters ---
parameters <- list(
    beta0 = rep(1, num_groups), beta1 = rep(1, num_groups), mu = c(3, 3), sigma = c(3, 3)
)

# fixed parameters
# map <- list(sigma = factor(NA), beta1 = rep(factor(NA), num_groups))

## create model object ---

obj <- MakeADFun(
    data = input,
    parameters = parameters,
    # map = map,
    random = "beta1",
    DLL = "logreg"
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
fit <- tmbstan(obj, chains = chains, cores = chains, init = unlist(parameters), iter = iter, 
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

plot_data <- data.frame(x = x[,2], y = y, group = factor(x[,3]))

yhat_post <- apply(
        mcmc_results,
        MARGIN = 1,
        FUN = \(x) obj$report(x)$yhat
    ) |>
    t()

plot_data$yhat <- apply(yhat_post, MARGIN = 2, FUN = median)
plot_data$yhat_lwr <- apply(yhat_post, MARGIN = 2, FUN = \(x) quantile(x, .025))
plot_data$yhat_upr <- apply(yhat_post, MARGIN = 2, FUN = \(x) quantile(x, .975))

ggplot(data = plot_data, aes(x = x)) +
    geom_point(aes(y = y, color = group)) +
    geom_line(aes(y = yhat, color = group)) +
    geom_ribbon(aes(ymin = yhat_lwr, ymax = yhat_upr, fill = group), alpha = .1) +
    theme_bw() +
    theme(legend.position = "none")

# test for overdispersion ---

phat_post <- apply(
        mcmc_results,
        MARGIN = 1,
        FUN = \(x) obj$report(x)$phat
    ) |>
    t()

phat <- apply(phat_post, MARGIN = 2, FUN = median)

chi_sq <- sum((y - size*phat)^2 / (size*phat*(1-phat)))

pchisq(chi_sq, n*num_groups - ncol(beta), lower.tail = FALSE)
