from scipy.stats import expon

def generate_data_rates(num_samples=int(1e6)):
  vid_data_rates_metis = []
  pkt_size = 1.66*2**20 # MB per second
  mean_reading_time = 0.033 # 33ms mean reading time
  for i in range(num_samples):
    # Draw a waiting time
    wait_time = expon.rvs(mean_reading_time)
    vid_data_rates_metis.append(pkt_size/wait_time)
  return vid_data_rates_metis
