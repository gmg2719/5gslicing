"""
This file contains functions for building a model of the load to be expected in
a cell of a given radius and ue density. The load is calculated assuming that the
UEs generate traffic demands following several distributions (e.g. web, video,
ftp), and the UE data rate is capped based on the distance from the BS.

There is also a function that allows aggregating traffic from multiple BSs and
calculating the 95th percentile of the maximum data rate. This is used to calculate
capacity requirements in the transport network.
"""

import numpy as np
from scipy.stats import norm, poisson

def path_loss_5g(ue_dist):
  """
  Based on Haneda et al 5G 3gpp-like channel models... VTC 2016
  The CI (Close-In) model.
  """
  # Calculate the probability of LoS
  p_los = np.minimum(18/ue_dist,1)*(1-np.exp(-ue_dist/63)+np.exp(-ue_dist/63))
  # Parameters of CI-model
  n = np.zeros_like(p_los)
  SF = np.zeros_like(p_los)
  # Path loss exp
  n[p_los >= 0.5] = 2.0 # High prob of LoS
  n[p_los < 0.5] = 3.0
  # Shadow-fading stddev
  SF[p_los >= 0.5] = 4.1 # High prob of LoS
  SF[p_los < 0.5] = 6.8
  fspl1m = 20*np.log10(4*np.pi*3.8*1e9/299792458)
  shadow = norm.rvs(scale=SF)
  return fspl1m + 10*n*np.log10(ue_dist) + shadow

# The following map of SINR to MCS from
#     Ramos et al, "Mapping between...", WPMC 2019
sinr_mcs_map = [
    -4.63,-3.615,-2.6,-1.36,-0.12,1.17,2.26,3.595,
    4.73,6.13,7.53,8.1,8.67,9.995,11.32,12.78,14.24,14.725,15.21,16.92,18.3,
    19.975,21.32,22.395,23.47,25.98,28.49,31.545,34.6
]

def get_mcs_index(sinr):
  """Returns the MCS index that is supported at the given SINR"""
  mcs = 0
  for s in sinr_mcs_map[1:]:
    if sinr < s: return mcs
    mcs += 1
  return len(sinr_mcs_map)-1

# Table 5.1.3.1-1 from 38.214
# Modulation order and target code rate x1024 for mcs index 
mcs_index_tbl = [
    (2,120),(2,157),(2,193),(2,251),(2,308),(2,379),(2,449),(2,526),(2,602),(2,679),
    (4,340),(4,378),(4,434),(4,490),(4,553),(4,616),(4,658),(6,438),(6,466),(6,517),
    (6,567),(6,616),(6,666),(6,719),(6,772),(6,822),(6,873),(6,910),(6,948)
]

def data_rate_5g(mod_ord, code_rate):
  """
  5G NR UE data rate, based on 38.306 4.1.2
  Assuming no carrier aggregation, 100MHz channel, 4 MIMO layers.
  """
  mu = 2 # Use FR1 (<6GHz), BW 100MHz so numerology (mu) = 2 (60KHz SCS)
  Tu = 1e-3/(14*2**mu) # average OFDM symbol duration
  N_prb = 132 # Based on 38.101-2 table 5.3.2-1
  oh = 0.14 # Overhead for FR1 DL
  layers = 4 # MIMO
  return 1e-6*layers*mod_ord*(code_rate/1024)*N_prb*12*(1-oh)/Tu

def get_ue_data_rate(dist):
  """
  Determine the maximum data rate to be expected at the given distance.

  First calculates the SINR using the path loss model, assuming 
  BS EIRP of 60dBm and UE EIRP of 40dBm (so 100dBm total gain).
  Then determines the MCS index from the SINR, and computes the data rate.
  """
  sinr = 100 - path_loss_5g(dist)
  if isinstance(dist, np.ndarray):
    return [data_rate_5g(*mcs_index_tbl[get_mcs_index(s)]) for s in sinr]
  else:
    return data_rate_5g(*mcs_index_tbl[get_mcs_index(sinr)])

# Sample user positions.
# From a Poisson Point Process on the circle with radius of the cell
def sample_users(user_density, cell_radius):
  """
  Creates a generator of user positions from a Poisson Point Process with
  lambda set to user_density.
  Returns the distance from the cell centre
  """
  lbda = np.pi*cell_radius**2*user_density
  num_users = poisson.rvs(lbda)
  for i in range(num_users):
    yield cell_radius*np.random.rand()
  return

def generate_load(user_density, cell_radius, traffic_types):
  """
  Generates an estimate of load in a cell of given radius and user density.
  Model:
  * each user will generate one of three traffic types (web, ftp, video)
  * the demand will be sampled from the data rates for the chosen traffic type
  * the demand will be capped by the maximum data rate permissible
  * the maximum data rate permissible for the ue will be estimated based on the
  distance from the antenna.
  The UEs are spread in the cell using a PPP.

  Parameters:
  user_density      -- UEs per km2
  cell_radius       -- in km
  traffic_types     -- list of traffic rate distributions (in Bps).

  Returns:
  Cell load in MBps and the number of UEs for the cell.
  """
  total_load = 0
  load = []
  traffic_types = [web_data_rates, ftp_data_rates, vid_data_rates_metis]
  num_users = 0
  for d in sample_users(user_density, cell_radius):
    # Sample a traffic model
    traffic_type = traffic_types[np.random.randint(0,3)]
    ue_demand = traffic_type[np.random.randint(0, len(traffic_type))]
    max_data_rate = get_ue_data_rate(d) # This is in Mbps so must be converted
    this_load = min(ue_demand*8, max_data_rate*2**20)
    total_load += this_load
    num_users += 1
  return total_load/(2**20), num_users

def aggregate_mux(k, cell_load, aggregate_mux_lookup=None):
  """
  Aggregates the traffic load of k cells and calculates the 95th percentile of
  the total load.
  Uses MC simulations, assuming independent load for the cells.
  Can be made faster using a lookup table.

  Parameters
  k   --  number of cells to aggregate
  cell_load   -- distribution of cell load. Can be None if the lookup table exists
  Returns
  Capacity required to maintain 95-th percentile of aggregatd load
  """
  if aggregate_mux_lookup is not None:
    if k % 10 != 1:
      k = int(np.ceil(k/10)*10)+1
    return aggregate_mux_lookup[k]
  if cell_load is None:
    raise ValueError('Lookup table is not defined so cell_load cannot be None!')
  total_load = []
  for s in range(int(1e4)):
    agg_load = sum(cell_load[np.random.randint(0,len(cell_load))]
                   for i in range(k))
    total_load.append(agg_load)
  total_load.sort()
  prob = np.arange(len(total_load))/len(total_load)
  p95 = np.argmin(np.abs(prob - 0.95))
  return total_load[p95] 
