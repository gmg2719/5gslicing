import util
from scipy.stats import expon

def generate_data_rates(num_samples=int(1e6)):
  web_data_rates = []
  for i in range(num_samples):
    # Draw a packet size from lognormal distr
    pkt_size = util.trunc_lognormal_sample(100, 2*2**20, 25032, 10710)
    # Draw a waiting time
    wait_time = expon.rvs(30)
    web_data_rates.append(pkt_size/wait_time)
  return web_data_rates
