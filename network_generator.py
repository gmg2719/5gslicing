import numpy as np
from scipy.stats import norm
import copy
from scipy.cluster.vq import kmeans2, whiten
from cell_load_model import aggregate_mux
import networkx as nx
import pickle

# Algorithm configuration
config = {
    'high_density': 0.6, # Ratio of population living in highly dense urban
    'urban': 0.39, # Ratio of population living in regular urban area
    'rural': 0.01, # Ratio of population living in rural area
    'isd_high': 0.2, # 200m high density ISD, per 3GPP
    'isd_urban': 0.5,
    'isd_rural': 1.7,
    'ue_density_high': 2500, # UEs per km2 in dense areas
    'ue_density_med': 400, # urban areas
    'ue_density_low': 100, # rural areas
    'prob_disagg_high': 0.8, # 80% of RUs in dense areas will be disaggregated
    'prob_disagg_urban': 0.3,
    'prob_disagg_rural': 0.05,
    'rate_fx_peak': 25000, # In Mbps
    'rate_f1_peak': 6000,  # In Mbps
    'rate_ngc_peak': 6000, # In Mbps
}

def get_latency(dist):
  """Transport latency through single link"""
  return dist/300000

def get_distance(a, b):
  """Computes the distance between two coordinates, provided as (x,y) tuples"""
  return np.sqrt((a[0]-b[0])**2+
                 (a[1]-b[1])**2)

class RU():
  """Represents a Radio Unit"""
  def __init__(self, coord):
    x, y = coord
    self.id = 'ru'+str(np.random.randint(0,100000))
    self.x = x
    self.y = y
    self.inbound_traffic = None

  def __repr__(self):
    return "({0.x},{0.y})".format(self)

class FCP():
  """Represents a Fibre Connection Point"""
  def __init__(self):
    self.id = 'fcp'+str(np.random.randint(0,100000))
    self.in_rate = None
    self.out_rate = None
    self.coords = None
    self.compute_cores = 0
    self.compute_storage = 0
    self.next_hop_latency = None # Distance to the next hop towards RAN

  def graph_attributes(self):
    return {'type': 'fcp',
            'cores': self.compute_cores,
            'storage': self.compute_storage,
            'traffic': (self.in_rate, self.out_rate)}

class AccessFCP(FCP):
  """Fibre Connection Point used in the access ring. Aggregates RUs."""
  def __init__(self, type):
    super().__init__()
    self.rus = []
    self.type = type # 1 for NG 2 for Fx
    self.next_hop_latency = get_latency(6.5)

  def graph_attributes(self):
    return {'type': 'sap',
            'tas': [ru.id for ru in self.rus],
            'traffic': (self.in_rate, self.out_rate)}


class EdgeFCP(FCP):
  """Fibre Connection Point used in the edge ring. Aggregates access rings."""
  def __init__(self):
    super().__init__()
    self.ring = None
    self.num_rus = None
    self.next_hop_latency = get_latency(5)

class CoreFCP(FCP):
  """Fibre Connection Point used in the core ring. Aggregates edge rings."""
  def __init__(self):
    super().__init__()
    self.ring = None
    self.num_rus = None
    self.next_hop_latency = get_latency(50)

class POP():
  """Point of Presence hosting compute resources."""
  def __init__(self):
    self.id = 'pop'+str(np.random.randint(0,100000))
    self.fcps = []
    self.in_rate = None
    self.compute_cores = 0
    self.compute_gpus = 0
    self.compute_storage = 0
    self.nfs = None # Network Functions

  def graph_attributes(self):
    return {'type': 'pop',
            'cores': self.compute_cores,
            'storage': self.compute_storage,
            'gpus': self.compute_gpus,
            'traffic': self.in_rate}

class Ring():
  """Transport network ring, connects set of PoPs"""
  def __init__(self):
    self.fcps = []
    self.pops = []
    self.num_rus = None
    self.pop2pop_latency = None
    self.pop2fcp_latency = None
    # Capacity is allocated for 100% of Fx + 95% of all NG
    self.capacity = None

class NwkInfrastructure():
  """Store the components of the 5G infrastructure"""
  def __init__(self):
    self.ru_cells = None
    # Access
    self.access_fcps = []  # aggregated RUs, type (2:disagg; 1:agg), input rate, output rate, coords
    self.access_pops = []  # FCPs, compute power, NFs
    self.access_rings = [] # FCPs, PoPs, capacity
    # Edge
    self.edge_fcps = []  # aggregated access rings, input rate, output rate, coords
    self.edge_pops = []  # FCPs, compute power, NFs
    self.edge_rings = [] # FCPs, PoPs, capacity
    # Core
    self.core_fcps = []
    self.core_ring = None

  def _add_fcp(self, fcp, g):
    """Add an fcp and the ring or RUs connected to the fcp."""
    # g.add_node(fcp.id, **fcp.graph_attributes())
    if isinstance(fcp, AccessFCP):
      g.add_node(fcp.id, **fcp.graph_attributes())
      return fcp
    else:
      # Add the ring attached to the fcp
      return self._add_ring_to_graph(fcp.ring, g)
      #g.add_edge(next_pop.id, parent_pop.id,
      #        **{'capacity': fcp.in_rate, 'latency':fcp.next_hop_latency})

  def _add_ring_to_graph(self, ring, g):
    # Add first pop in ring
    g.add_node(ring.pops[0].id, **ring.pops[0].graph_attributes())
    # Add fcps
    for f in ring.pops[0].fcps:
      next_pop = self._add_fcp(f, g)
      g.add_edge(ring.pops[0].id, next_pop.id,
              **{'capacity': f.out_rate,
                  'latency':ring.pop2fcp_latency+f.next_hop_latency})
    if len(ring.pops) == 1: return
    # Add the rest of the pops
    for p1, p2 in zip(ring.pops[:-1], ring.pops[1:]):
      # Add nodes with their attributes
      g.add_node(p2.id, **p2.graph_attributes())
      g.add_edge(p1.id, p2.id, **{'capacity': ring.capacity, 'latency':ring.pop2pop_latency})
      # Add fcps
      for f in p2.fcps:
        next_pop = self._add_fcp(f, g)
        g.add_edge(p2.id, next_pop.id,
                **{'capacity': f.out_rate,
                    'latency':ring.pop2fcp_latency+f.next_hop_latency})
    # Complete the ring
    g.add_edge(ring.pops[-1].id, ring.pops[0].id,
               **{'capacity': ring.capacity, 'latency':ring.pop2pop_latency})
    return ring.pops[0]

  def generate_nx_graph(self):
    """Represent the infrastructure as a NetworkX graph"""
    g = nx.Graph()
    self._add_ring_to_graph(self.core_ring, g)
    return g

class NetworkGenerator():

  def __init__(self, area_width, area_cell_width=0.1, aggregate_mux_lookup=None):
    """
    Create a 5G infrastructure generator.

    Parameters:

    area_width          -- the width of the square area, in kms
    area_cell_width     -- the area is broken in small cells (in kms)
    aggregate_mux_lookup-- pickle file object containing lookup table for aggregation.
    """
    # Define the area as a matrix of cells of configured width
    row_cells = int(np.ceil(area_width/area_cell_width))
    # Need to round up to next square
    self.num_cells = int(row_cells**2)
    area_cells = np.array(range(self.num_cells))
    self.area_cells = area_cells.reshape(row_cells, row_cells)
    self.area_cell_width = area_cell_width
    # List of populations deployed
    self.populations = []
    # Population density in each cell
    self.population_density = np.zeros_like(self.area_cells, dtype=np.float64)
    if aggregate_mux_lookup is None:
      try:
          aggregate_mux_lookup = open('aggregate_mux_lookup', 'rb')
      except IOError:
          print("Aggregate mux lookup not ready yet.")
          return
    self.aggregate_mux_lookup = pickle.load(aggregate_mux_lookup)

  def dist_from_pdf_centre(self, pdf_centre):
    """
    Returns the distance (kms) from x,y to the PDF centre.

    Parameters:

    pdf_centre  -- (x,y) coordinates.
    """
    cell_col = self.area_cells % np.sqrt(self.num_cells)
    cell_row = self.area_cells // np.sqrt(self.num_cells)
    cell_x = 0.5*(2*cell_col+1)*self.area_cell_width
    cell_y = 0.5*(2*cell_row+1)*self.area_cell_width
    c_x, c_y = pdf_centre
    return np.sqrt((c_x-cell_x)**2 + (c_y-cell_y)**2)

  def generate_demographic(self, centre, pdf_width):
    """
    Add a population distribution over the deployment area.
    The population is a 2D normal distribution with standard deviation (pdf_width)
    expressed in kms, and zero mean.
    For each cell of the deployment area the population density will be estimated
    and classified as high (urban), common (suburban), and rural.

    Returns:

    matrix with each cell containing its index
    matrix with each cell containing the population density
    """
   # Compute the PDF for each cell, based on its distance from the centre
    pdfs = norm.pdf(self.dist_from_pdf_centre(centre), 0, pdf_width)
    # Compute the thresholds for population density, based on Quantile function
    # --> P(population density > Threshold) is CDF(x) = Threshold
    # --> we want the threshold so CDF-1(x) which is quantile function
    ppf_high = norm.ppf(config['high_density'], 0, pdf_width)
    ppf_urban = abs(norm.ppf(config['high_density'] + config['urban'],
                                0, pdf_width))
    # Then convert those values into PDFs
    thresh_high = norm.pdf(ppf_high, 0, pdf_width)
    thresh_urban = norm.pdf(ppf_urban, 0, pdf_width)
    # Discretize the PDF based on the thresholds
    pdfs[pdfs >= thresh_high] = 3
    pdfs[(pdfs >= thresh_urban) & (pdfs < 1)] = 2
    pdfs[pdfs < thresh_urban] = 1
    # Add the PDF values to the deployment area
    self.population_density += pdfs

  def _deploy_rus(self):
    """
    Deploys RUs based on the ISD defined in the configuration.
    Returns a matrix of same shape as area cells where each element represents
    an area cell and is 1 if an RU is deployed, otherwise 0.
    """
    ru_cells = np.zeros_like(self.area_cells)
    # Deploy in cells >= rural with rural ISD
    isd_rural_cells = int(config['isd_rural']/self.area_cell_width)
    ru_cells[0:len(self.area_cells):isd_rural_cells,
           np.arange(0,len(self.area_cells), isd_rural_cells),] = 1
    # Clear the urban areas and higher
    ru_cells[self.population_density >= 2] = 0
    # Deploy
    isd_urban_cells = int(config['isd_urban']/self.area_cell_width)
    tmp_urban_base = self.population_density >= 2
    tmp_urban_cells = np.zeros_like(self.area_cells)
    tmp_urban_cells[0:len(self.area_cells):isd_urban_cells,
            np.arange(0, len(self.area_cells), isd_urban_cells)] = 1
    ru_cells[np.array((tmp_urban_base & tmp_urban_cells), dtype=np.bool)] = 1
    # Clear the high density areas
    ru_cells[self.population_density == 3] = 0
    # Deploy
    isd_high_cells = int(config['isd_high']/self.area_cell_width)
    tmp_high_base = self.population_density == 3
    tmp_high_cells = np.zeros_like(self.area_cells)
    tmp_high_cells[0:len(self.area_cells):isd_high_cells,
            np.arange(0, len(self.area_cells), isd_high_cells)] = 1
    ru_cells[np.array((tmp_high_base & tmp_high_cells), dtype=np.bool)] = 1
    return ru_cells

  def generate_5g_infra(self, random_seed):
    # RNG
    rng = np.random.default_rng(random_seed)
    # Store the infrastructure elements
    infra = NwkInfrastructure()
    infra.ru_cells = self._deploy_rus()
    pdfs = self.population_density
    #------------------
    # First step is to group the RUs into access FCPs
    isd = {1:config['isd_rural'], 2:config['isd_urban'], 3:config['isd_high']}
    prob_disag = {1:config['prob_disagg_rural'],
                  2:config['prob_disagg_urban'],
                  3:config['prob_disagg_high']}
    for density in [1,2,3]:
      # Make a copy of the RU matrix
      tmp = copy.deepcopy(infra.ru_cells)
      # Discard RUs from other density zones
      tmp[pdfs != density] = 0
      # Process the area by overlaying squares that contain 36 RUs
      # We call this a site
      # Start in the top left corner of the area
      top_y = np.argwhere(np.count_nonzero(pdfs == density, 1))[0][0]
      top_x = np.argwhere(np.count_nonzero(pdfs == density, 0))[0][0]
      # Also find the bottom right corner
      bot_y = np.argwhere(np.count_nonzero(pdfs == density, 1))[-1][0]
      bot_x = np.argwhere(np.count_nonzero(pdfs == density, 0))[-1][0]
      # Determine the width of the site in area cells
      site_width = int(np.ceil(6*isd[density]/self.area_cell_width))
      # Process the squares
      for x_c in range(top_x, bot_x, int(site_width)):
        for y_c in range(top_y, bot_y, int(site_width)):
          fcp_type = 2 if rng.random() < prob_disag[density] else 1
          fcp = AccessFCP(fcp_type)
          # TODO Should have provision in case the number of RUs is too low to aggregate
          num_rus = np.count_nonzero(tmp[x_c:x_c+site_width, y_c:y_c+site_width])
          # Get the coordinates of the RUs
          fcp.rus = list(map(RU, np.argwhere(tmp[x_c:x_c+site_width, y_c:y_c+site_width]) + [x_c,y_c]))
          if fcp_type == 2:
            # For disaggregated, this is Fx interface
            fcp.in_rate = config['rate_fx_peak']
            fcp.out_rate = num_rus * fcp.in_rate
          elif fcp_type == 1:
            # For aggregated RAN, this is F1 or NGC and we can apply mux gain
            fcp.in_rate = config['rate_ngc_peak']
            fcp.out_rate = aggregate_mux(num_rus, None, self.aggregate_mux_lookup)
            # For half of these FCPs we allocate 100 threads
            fcp.compute_threads = 100 if rng.random() >= 0.5 else 0
          fcp.coords = (int(x_c+site_width/2), int(y_c+site_width/2))
          infra.access_fcps.append(fcp)
    #------------------
    # Group the access FCPs into access rings and allocate PoPs
    # Sort all the access fcps lexicographically (first X, then Y coords)
    tmp = copy.deepcopy(infra.access_fcps)
    tmp.sort(key=lambda f: f.coords[0])
    while len(tmp) > 0:
      fcp = tmp[0] # This is the reference
      tmp = tmp[1:]
      # Sort the rest of the fcps in order of their distance from the reference
      ring = Ring()
      closest_fcps = sorted(tmp, key=lambda x: get_distance(fcp.coords,x.coords))
      ring.fcps = [fcp]+closest_fcps[:15]
      # Determine required capacity for the access ring
      # This will be 100% of Fx RUs + 95% of cumulated NG RUs
      num_fx_rus = 0
      num_ng_rus = 0
      for f in ring.fcps:
        if f.type == 2: num_fx_rus += len(f.rus)
        elif f.type == 1: num_ng_rus += len(f.rus)
      ring.capacity = num_fx_rus*config['rate_fx_peak'] \
                      + aggregate_mux(num_ng_rus, None, self.aggregate_mux_lookup)
      ring.num_rus = num_fx_rus + num_ng_rus
      # Allocate 4 PoPs with compute
      ring.pops = [POP() for i in range(4)]
      fcps_per_pop = int(np.ceil(len(ring.fcps)/len(ring.pops)))
      for idx,p in enumerate(ring.pops):
        p.fcps = ring.fcps[idx*fcps_per_pop:min(len(ring.fcps),(idx+1)*fcps_per_pop)]
        p.compute_cores = 400 if rng.random() >= 0.5 else 0
        p.compute_gpus = 20 if rng.random() >= 0.5 else 0
      # One of the PoPs will host the primary video cache of 25TB
      rng.choice(ring.pops).compute_storage = 25000
      # Remove the fcps from the list
      for f in closest_fcps[:15]:
        tmp.remove(f)
      # Set the latency measures for the ring
      ring.pop2fcp_latency = get_latency(5)
      ring.pop2pop_latency = get_latency(8)
      infra.access_rings.append(ring)
    #------------------
    # Create the Edge FCPs
    for ar in infra.access_rings:
      ar_centre_x = sum(f.coords[0] for f in ar.fcps)/len(ar.fcps)
      ar_centre_y = sum(f.coords[1] for f in ar.fcps)/len(ar.fcps)
      fcp = EdgeFCP()
      fcp.ring = ar
      fcp.coords = (ar_centre_x, ar_centre_y)
      fcp.num_rus = ar.num_rus
      # The output rate of the FCP can take into account statistical muxing because
      # the DUs are in the access ring
      fcp.out_rate = aggregate_mux(fcp.num_rus, None, self.aggregate_mux_lookup)
      infra.edge_fcps.append(fcp)
    # Join the edge FCPs into edge rings
    fcps_per_edge_ring = 5
    if len(infra.edge_fcps) % fcps_per_edge_ring == 0:
      num_edge_rings = len(infra.edge_fcps)/fcps_per_edge_ring
    else:
      num_edge_rings = int(np.ceil(len(infra.edge_fcps)/fcps_per_edge_ring))
      fcps_per_edge_ring = int(np.ceil(len(infra.edge_fcps)/num_edge_rings))
    # Use k-means clustering to find the coordinates of the CoreFCPs.
    # They will determine which edge FCPs form edge rings
    whitened = [fcp.coords for fcp in infra.edge_fcps]
    centroids, _ = kmeans2(whitened, num_edge_rings)
    # Each centroid is connected to the closest edge fcps
    rem = len(infra.edge_fcps)
    for ring_centre in centroids:
      to_add = min(rem, fcps_per_edge_ring)
      # find the FCPs that are connected in this ring
      fcps = sorted(infra.edge_fcps,
                  key=lambda x: get_distance(ring_centre, x.coords))[:to_add]
      edge_ring = Ring()
      edge_ring.fcps = fcps
      rem -= to_add
      edge_ring.num_rus = sum(f.num_rus for f in fcps)
      edge_ring.capacity = aggregate_mux(min(1991, edge_ring.num_rus), None,
                                            self.aggregate_mux_lookup)
      infra.edge_rings.append(edge_ring)
      # One of the edge PoPs will have 1000 threads and 100TB
      edge_ring.pops = [POP(), POP()]
      edge_ring.pops[0].fcps = edge_ring.fcps[:3]
      edge_ring.pops[1].fcps = edge_ring.fcps[3:]
      # Set latency measures for the edge ring
      edge_ring.pop2fcp_latency = get_latency(20)
      edge_ring.pop2pop_latency = get_latency(30)
      p = rng.choice(edge_ring.pops)
      p.compute_cores = 1000
      p.compute_storage = 100000
      # Use the ring_centre as the core FCP
      core_fcp = CoreFCP()
      core_fcp.ring = edge_ring
      core_fcp.in_rate = ring.capacity
      core_fcp.out_rate = ring.capacity
      core_fcp.num_rus = edge_ring.num_rus
      core_fcp.coords = ring_centre
      infra.core_fcps.append(core_fcp)
    # Finally create the core ring
    infra.core_ring = Ring()
    infra.core_ring.fcps = infra.core_fcps
    infra.core_ring.num_rus = sum(f.num_rus for f in infra.core_fcps)
    infra.core_ring.capacity = aggregate_mux(min(1991, edge_ring.num_rus), None,
                                            self.aggregate_mux_lookup)
    infra.core_ring.pop2fcp_latency = get_latency(100)
    infra.core_ring.pop2pop_latency = get_latency(300)
    # Allocate PoPs in the core
    per_core = 5000 # in Mbps 
    core_dc = POP()
    core_dc.compute_cores = np.ceil(infra.core_ring.capacity / per_core)
    infra.core_ring.pops = [core_dc]
    core_dc.fcps = infra.core_ring.fcps
    return infra
