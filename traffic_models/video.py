import util

def generate_data_rates(num_samples=int(1e6)):
  min_vid_pkt = 560/8*2**10 # 560KBs #3.32*2**20  # 3.32 MB
  max_vid_pkt = 4.2/8*2**20 # 4.2MBs # 20.75*2**20 # 20.75 MB
  vid_data_rates = []
  for i in range(num_samples):
    # Draw a packet size from lognormal distr
    pkt_size = bounded_pareto_sample(min_vid_pkt, max_vid_pkt, 1.67)
    # Draw a waiting time
    wait_time = bounded_pareto_sample(0.832*1e-3, 5.2*1e-3, 1.67)
    vid_data_rates.append(pkt_size/wait_time)
  return vid_data_rates
