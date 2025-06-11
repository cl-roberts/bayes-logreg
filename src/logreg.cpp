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

  // priors
  DATA_VECTOR(beta1_prior); 
  DATA_VECTOR(beta2_prior);

  // parameters
  // PARAMETER(p);
  PARAMETER_VECTOR(beta);
  
  // model objects
  int n = y.size();
  vector<Type> logit_phat(n);
  vector<Type> phat(n);
  vector<Type> yhat(n);

  // model
  for (int i = 0; i < n; i++) {
    logit_phat(i) = beta(0) + beta(1)*x(i);
    phat(i) = 1 / (1 + exp(-logit_phat(i)));
    yhat(i) = size(i) * phat(i);
  }
  
  // objective function
  Type negLogLik = 0.0;

  for (int i = 0; i < n; i++) {
    negLogLik -= y(i) * logit_phat(i);
    negLogLik += size(i)*log(1 + exp(logit_phat(i)));
  }

  // priors
  negLogLik -= dnorm(beta(0), beta1_prior(0), beta1_prior(1), true);
  negLogLik -= dnorm(beta(1), beta2_prior(0), beta2_prior(1), true);
  
  // report
  REPORT(yhat);
  REPORT(phat);
  
  return negLogLik;
  
}
