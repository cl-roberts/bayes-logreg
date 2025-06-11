//----------------------------------------------------------------------------//

// Bayesian logistic regression example in TMB

// cpp file

// CL Roberts

// 6/11/2025

//----------------------------------------------------------------------------//

// libraries
#include <TMB.hpp>


// model
template<class Type>
Type objective_function<Type>::operator() ()
{

  // data
  DATA_VECTOR(y);
  DATA_VECTOR(x);
  DATA_VECTOR(size);
  DATA_FACTOR(group);

  // priors
  DATA_VECTOR(beta0_prior);    
  DATA_VECTOR(beta1_prior);
  DATA_VECTOR(mu0_prior); 
  DATA_VECTOR(sigma0_prior); 
  DATA_VECTOR(mu1_prior); 
  DATA_VECTOR(sigma1_prior); 

  // parameters
  PARAMETER_VECTOR(beta0);
  PARAMETER_VECTOR(beta1);
  PARAMETER_VECTOR(mu);
  PARAMETER_VECTOR(sigma);
  
  // model objects
  int n = y.size();
  int num_groups = beta1.size();
  int j;

  vector<Type> logit_phat(n);
  vector<Type> phat(n);
  vector<Type> yhat(n);
  
  // objective function
  Type negLogLik = 0.0;

  // random effect
  for (int j = 0; j < num_groups; j++) {
    negLogLik -= dnorm(beta0(j), mu(0), sigma(0), true);
    negLogLik -= dnorm(beta1(j), mu(1), sigma(1), true);
  }

  // model
  for (int i = 0; i < n; i++) {
    j = group(i)-1;
    logit_phat(i) = beta0(j) + beta1(j)*x(i);
    phat(i) = 1 / (1 + exp(-logit_phat(i)));
    yhat(i) = size(i) * phat(i);
  }  
  
  // likelihood
  for (int i = 0; i < n; i++) {
    negLogLik -= y(i) * logit_phat(i);
    negLogLik += size(i)*log(1 + exp(logit_phat(i)));
  }

  // priors
  for (int j = 0; j < num_groups; j++) {
    negLogLik -= dnorm(beta0(j), beta0_prior(0), beta0_prior(1), true);
    negLogLik -= dnorm(beta1(j), beta1_prior(0), beta1_prior(1), true);
  }
  negLogLik -= dnorm(mu(0), mu0_prior(0), mu0_prior(1), true);
  negLogLik -= dnorm(sigma(0), sigma0_prior(0), sigma0_prior(1), true);
  negLogLik -= dnorm(mu(1), mu1_prior(0), mu1_prior(1), true);
  negLogLik -= dnorm(sigma(1), sigma1_prior(0), sigma1_prior(1), true);
  
  // report
  REPORT(yhat);
  REPORT(phat);
  
  return negLogLik;
  
}
