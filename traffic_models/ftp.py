import util
from scipy.stats import expon

def generate_data_rates(num_samples=int(1e6)):
  ftp_data_rates = []
  for i in range(num_samples):
    # Draw a packet size from lognormal distr
    pkt_size = util.trunc_lognormal_sample(1, 5*2**20, 2**20*0.722, 2*2**20)
    # Draw a waiting time
    wait_time = expon.rvs(180)
    ftp_data_rates.append(pkt_size/wait_time)
  return ftp_data_rates
