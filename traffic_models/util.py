import numpy as np
from scipy.stats import norm
from scipy.special import erf

def lognormal_cdf(x, sigma, mu):
  return 0.5+0.5*erf((np.log(x)-mu)/(np.sqrt(2)*sigma))

# Truncated log-normal sampling
def trunc_lognormal_sample(a,b,mu,sigma):
  lncdf_a = lognormal_cdf(a,sigma,mu)
  lncdf_b = lognormal_cdf(b,sigma,mu)
  cdf = (lncdf_b - lncdf_a)*np.random.random() + lncdf_a
  ppf_norm = norm.ppf(cdf, mu, sigma)
  return np.exp(ppf_norm)

def trunc_pareto_pdf(x, lo, hi, alpha):
  return alpha*lo**alpha*(x**(-alpha-1))/(1-(lo/hi)**alpha)

def pareto_cdf(x, m, alpha):
  return 1 - (m/x)**alpha

def bounded_pareto_sample(lo, hi, alpha):
  x = np.random.random()
  return (-(x*hi**alpha-x*lo**alpha-hi**alpha)/(hi**alpha*lo**alpha))**(-1/alpha)
