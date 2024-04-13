'''
Functions for part 4 of main HNSCC image analysis pipeline: Spatial processing
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''

##FUNCTIONS

def power_set(items, conds):

	"""
	Generates the power set of a given set of items, excluding the empty set and including only subsets 
	that contain a specified set of conditions.

	Parameters
	----------
	items : list
		A list of items for which the power set is to be generated.
	conds : set
		A set of conditions that must be included in the subsets for them to be added to the output list.

	Returns
	-------
	out_sets : list of set
		A list of sets, each being a subset of `items` that also includes all elements of `conds`.
	"""

	N = len(items)
	out_sets = []

	#Enumerate the 2 ** N possible combinations
	for i in range(2 ** N):
		
		combo = set([])
		for j in range(N):
			#Test bit jth of integer i
			if (i >> j) % 2 == 1:
				
				combo = combo.union(set([items[j]]))
		
		if conds.issubset(combo):
			out_sets.append(combo)

	return out_sets


def power_set_alt(items, conds):
	
	"""
	Generates the power set of a given set of items, excluding the empty set, and including only subsets 
	that have at least one item in common with a specified set of conditions.

	Parameters
	----------
	items : list
		A list of items for which the power set is to be generated.
	conds : set
		A set of conditions used to filter the subsets. A subset is included in the output if it has 
		at least one common element with `conds`.

	Returns
	-------
	out_sets : list of set
		A list of sets, each being a subset of `items` that shares at least one element with `conds`.
	"""

	N = len(items)
	out_sets = []
	
	#Enumerate the 2 ** N possible combinations
	for i in range(2 ** N):
		
		combo = set([])
		for j in range(N):
			#Test bit jth of integer i
			if (i >> j) % 2 == 1:
				combo = combo.union(set([items[j]]))
		
		overlap = [m for m in conds if m in combo]
		if len(overlap) != 0:
			out_sets.append(combo)
	
	return out_sets


def prepare(outdir):
	
	"""
	Prepares and saves a comprehensive list of metrics for spatial analysis based on unique cell type combinations
	and various spatial metrics. This includes base metrics (like cell density), TLS counts and cellular neighborhood metrics.

	Parameters
	----------
	outdir : str
		The output directory where the metrics list will be saved.

	Returns
	-------
	Full_metrics : list
		A complete list of all metrics generated for spatial analysis.

	 Notes
	-----
	- Not all metrics in Full_metrics were calculated and included in the final analysis (CSR ratios for example)
	"""
	
	#Dictionary mapping major cell types to their subtype markers
	combs = {'Tumor': ['KI67', 'PDL1'],
			 'DC3': ['KI67', 'CD163', 'PDL1'], 
			 'CD4-Tcell': ['KI67', 'TCF1'], 
			 'Treg': ['KI67'],
			 'CD8-Tcell': ['KI67', 'TCF1'],
			 'Neutrophil': ['KI67','CD163', 'PDL1'], 
			 'Macrophage/Monocyte': ['KI67', 'CD163', 'PDL1']}
	
	#List of major celltypes
	celltypes_base = ['Tumor',
					 'DC3', 
					 'CD4-Tcell',
					 'Treg',
					 'CD8-Tcell', 
					 'Neutrophil', 
					 'Macrophage/Monocyte']
	
	celltypes = celltypes_base.copy()
	
	#Create list of all possible cell subtypes
	for ct in celltypes_base:
		for sub in combs[ct]:
			celltypes.append('_'.join([ct, sub+'pos']))
			celltypes.append('_'.join([ct, sub+'neg']))
	  
	#List of base metric types
	base_metrics = ['median_PDL1_expr_total', 'median_PDL1_expr_stroma', 'median_PDL1_expr_tumor', 'median_PDL1_expr_boundary_all', 'median_PDL1_expr_boundary_in', 'median_PDL1_expr_boundary_out',
				   'density_total', 'density_stroma', 'density_tumor', 'density_boundary_all', 'density_boundary_in', 'density_boundary_out',
				   'frequency_total_Comp', 'frequency_stroma_Comp', 'frequency_tumor_Comp',  'frequency_boundary_Comp_all', 'frequency_boundary_Comp_in', 'frequency_boundary_Comp_out',
				   'frequency_total_allCell', 'frequency_stroma_allCell', 'frequency_tumor_allCell', 'frequency_boundary_allCell_all', 'frequency_boundary_allCell_in', 'frequency_boundary_allCell_out']

	
	#Take the set cross product for a final list of base metrics applied to all cell types
	metric_list = ['&'.join([i1, i2]) for i1 in base_metrics for i2 in celltypes]
	
	#Additional metrics
	TLS_metrics = ['FullTLS', 'PartialTLS', 'DC_clusters', 'DC_cluster_prop', 'Processed_portion', 'Full_area', 'Tumor_area', 'BI_area', 'BO_area']
	
	#Median boundary distances (not used in final analysis)
	Boundary_dist_metrics = ['&'.join([comp, ct]) for comp in ['Tumor_dist', 'Stroma_dist'] for ct in celltypes[1:]]
	
	#Neighborhood metrics (CSR ratio not used in final analysis)
	Celltype_cross_prod = ['X'.join([ct1, ct2]) for ct1 in celltypes for ct2 in celltypes]
	Celltype_dist_metrics = ['&'.join([Measure, pair]) for Measure in ['Neighbor_ratio', 'CSR_ratio'] for pair in Celltype_cross_prod]
	
	#Assemble full metrics list
	Full_metrics = [metric_list, TLS_metrics, Boundary_dist_metrics, Celltype_dist_metrics]
	Full_metrics = [item for sublist in Full_metrics for item in sublist]
	
	#Save
	with open(f'{outdir}/Metrics_list', 'wb') as file:
		pickle.dump(Full_metrics, file)
	
	return Full_metrics


def get_TLS_count(data, FOV, Bcell_dist = 30*um_to_pixel, minBcell_cluster_size = 5, degree_of_buffering = 10*um_to_pixel, granularity = 0):  

	"""
	Calculates the count of partial and full tertiary lymphoid structures (TLS) based on B cell clusters of sufficient size,
	and the presence of dendritic cells (DC3) and T cells within these clusters.

	Parameters
	----------
	data : DataFrame
		The data containing cell identities and their centroid coordinates.
	FOV : str
		The field of view identifier.
	Bcell_dist : float, optional
		The distance threshold for considering B cells as part of the same cluster, in pixels. Defaults to 30*um_to_pixel.
	minBcell_cluster_size : int, optional
		The minimum number of B cells to consider a cluster as potential TLS. Defaults to 5.
	degree_of_buffering : float, optional
		The distance for the buffering around the cluster boundary for considering inclusion of other cell types, in pixels. Defaults to 10um.
	granularity : int, optional
		The granularity parameter for alphashape calculation, controlling the concavity of the cluster shape. Defaults to 0.

	Returns
	-------
	part_TLS : int
		The count of partial TLS, which have B cell clusters but lack either DC3 or sufficient T cells.
	full_TLS : int
		The count of full TLS, which have B cell clusters with at least one DC3 and more than one T cell.
	out_data : DataFrame
		A dataframe describing each identified TLS cluster with its type, centroid coordinates, original and padded areas, and B cell cluster size.
	"""  

	#Initialize outputs
	part_TLS = 0
	full_TLS = 0
	
	#Generate B cell clusters
	b_cell_data = data.loc[data['identity'] == 'Bcell', ['unique_ident', 'identity', 'centroid_x', 'centroid_y']].reset_index(drop = False, inplace = False)
	
	loc_matrix = b_cell_data[['centroid_x', 'centroid_y']].to_numpy()
	connectivity_matrix = scipy.spatial.distance_matrix(loc_matrix, loc_matrix, p=2)
	
	connectivity_matrix[connectivity_matrix < Bcell_dist] = 1
	connectivity_matrix[connectivity_matrix >= Bcell_dist] = 0
	
	#B cell graph
	total_graph = sparse.csr_matrix(connectivity_matrix)
	
	pot_TLS = []
	names = ['FOV', 'type', 'centroid_x', 'centroid_y', 'orig_area', 'padded_area', 'Bcell_cluster_size']
	out_data = pd.DataFrame(columns=names)
	
	#Add a set to store visited nodes
	visited_nodes = set() 

	#Traverse graph from all possible start nodes
	for start_node in range(0, len(loc_matrix)):
		
		#Check if node has been visited before
		if start_node in visited_nodes: 
			continue

		#BFS for completeness (memory usage seems ok)
		conn_graph = sparse.csgraph.breadth_first_order(total_graph, start_node, return_predecessors = False)    
		nodes_list = set(conn_graph)

		#Update visited_nodes set with all nodes in the cluster
		visited_nodes.update(nodes_list) 
		
		#Get all B cell clusters of sufficient size
		if len(nodes_list) > minBcell_cluster_size:

				#Create B cell cluster alphashape
				pot_TLS.append(nodes_list)
				points = loc_matrix[conn_graph, : ]
				TLS_circ = alphashape.alphashape(points, alpha=granularity)
				TLS_circ = make_valid(TLS_circ)
				orig_area = TLS_circ.area / (um_to_pixel**2)    # conversion to um
				padded_poly = TLS_circ.buffer(distance = degree_of_buffering)

				#Extract information
				centroid_x = TLS_circ.centroid.x
				centroid_y = TLS_circ.centroid.y
				padded_area = padded_poly.area / (um_to_pixel**2)
				cluster_size = len(nodes_list)

				#Collect DC3 and T cell locations
				DC_loc = data.loc[data['identity'] == 'DC3', ['centroid_x', 'centroid_y']].to_numpy()
				Tcell_loc = data.loc[data['identity'].str.contains('Tcell'), ['centroid_x', 'centroid_y']].to_numpy()
					
				outer_poly_coords = path.Path(padded_poly.boundary.coords)
				
				#Check for contained DC3s and T cells   
				contained_DC = np.array(outer_poly_coords.contains_points(DC_loc))
				contained_Tcell = np.array(outer_poly_coords.contains_points(Tcell_loc))
				DC_count = contained_DC[contained_DC == True].size
				Tcell_count = contained_Tcell[contained_Tcell == True].size

				#Add +1 to TLS counts depending on pre-set conditions
				if DC_count > 0 and Tcell_count > 1:
					full_TLS += 1 
					Type = 'full'
				else:
					part_TLS += 1
					Type = 'partial'

				out_Series = pd.Series([FOV, Type, centroid_x, centroid_y, orig_area, padded_area, cluster_size], index=names)
				out_data = pd.concat([out_data, out_Series.to_frame().T])
	
	return part_TLS, full_TLS, out_data


def get_Tumor_area(data, FOV, sample,
				   im_path, 
				   save_dir, 
				   cell_dist = 15*um_to_pixel, 
				   min_cluster_size = 10, 
				   degree_of_buffering = 8*um_to_pixel, 
				   granularity = 0.008):
	
	"""
	Identifies and visualizes tumor cell clusters from imaging data, applying spatial clustering and using alphashapes
	to define tumor areas and their boundaries. The resulting tumor areas are visualized on reduced resolution panCK images
	with scaling and buffering for QC purposes.

	Parameters
	----------
	data : DataFrame
		The data containing cell identities and their centroid coordinates.
	FOV : str
		The field of view identifier.
	sample : str
		The sample identifier.
	im_path : str
		The path to the panCK image file.
	save_dir : str
		The directory where the output images will be saved.
	cell_dist : float, optional
		The distance threshold for considering tumor cells as part of the same cluster, in pixels. Defaults to 15um.
	min_cluster_size : int, optional
		The minimum number of cells to consider a cluster as a tumor area. Defaults to 10.
	degree_of_buffering : float, optional
		The distance for the buffering around the cluster boundary, in pixels. Defaults to 8um.
	granularity : float, optional
		The granularity parameter for alphashape calculation, controlling the concavity of the cluster shape. Defaults to 0.008.

	Returns
	-------
	Tumor : list
		A list of buffered polygon objects representing tumor areas.
	Tumor_nodes : list
		A list of sets, each containing the indices of cells in a tumor cluster.
	fig : matplotlib.figure.Figure
		The figure object for the panCK image with tumor areas highlighted.
	ax : matplotlib.axes._subplots.AxesSubplot
		The axes object for the figure.
	out_im_name1 : str
		The file path for saving the output image with tumor areas visualized.

	Notes
	-----
	- This function loads an image, identifies tumor cell clusters based on spatial proximity, and visualizes these clusters
	  by applying alphashape to determine the cluster boundaries.
	- The visualization is scaled down for reduced memory usage.
	"""

	#Prepare out_im filename
	im_name = im_path.split('/')[-1].split('_')
	
	#Load the panCK image into a numpy array for the QC plot
	im = np.array(Img.open(im_path))
	fig, ax = plt.subplots(dpi=300)
	ax.imshow(im, cmap='gray')
	
	#Generate Tumor cell clusters
	cell_data = data.loc[data['identity'] == 'Tumor', ['unique_ident', 'identity', 'centroid_x', 'centroid_y']].reset_index(drop = False, inplace = False)
	loc_matrix = cell_data[['centroid_x', 'centroid_y']].to_numpy()
	connectivity_matrix = scipy.spatial.distance_matrix(loc_matrix, loc_matrix, p=2)
	connectivity_matrix[connectivity_matrix < cell_dist] = 1
	connectivity_matrix[connectivity_matrix >= cell_dist] = 0

	#Tumor graph
	total_graph = sparse.csr_matrix(connectivity_matrix)
	
	Tumor_nodes = []
	Tumor = []
	
	fig2, ax2 = 'a', 'b' #deepcopy((fig, ax))

	#Add a set to store visited nodes
	visited_nodes = set() 

	#Traverse graph from all possible start nodes
	for start_node in range(0, len(loc_matrix)):
		
		#Check if node has been visited before
		if start_node in visited_nodes: 
			continue

		#BFS for completeness (memory usage seems ok)
		conn_graph = sparse.csgraph.breadth_first_order(total_graph, start_node, return_predecessors = False)    
		nodes_list = set(conn_graph)

		#Update visited_nodes set with all nodes in the cluster
		visited_nodes.update(nodes_list) 
		
		#Get all Tumor cell clusters of sufficient size
		if len(nodes_list) > min_cluster_size:

				#Create tumor cell cluster alphashape
				Tumor_nodes.append(nodes_list)
				points = loc_matrix[conn_graph, : ]
				Tumor_circ = alphashape.alphashape(points, alpha=granularity)
				Tumor_circ = make_valid(Tumor_circ)
				padded_poly = Tumor_circ.buffer(distance = degree_of_buffering)
				
				#Add new tumor area/nest
				Tumor.append(padded_poly)
				
				#Scale the points and the polygon by 1/3 for displaying in reduced resolution images
				scaled_points = [(x * (1/3), y * (1/3)) for x, y in points]
				scaled_poly = scale(padded_poly, xfact=1/3, yfact=1/3, origin=(0, 0))

				#Add polygon patch to QC lot
				#Commented out in all instances for reproducible run due to necessary changes to descartes package source code
				#patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.22, color='tab:orange')
				#ax.add_patch(patch)
	
	#Create QC image filename for saving in downstream function
	out_im_name1 = f'{save_dir}/low_scale_circling/{sample}/Tumor_bounds_{im_name[1]}_{FOV}.png'
	
	return Tumor, Tumor_nodes, fig, ax, out_im_name1


def swap_xy(polygon):

	"""
	Swaps the x and y coordinates of a given polygon or multipolygon. This function is useful for visualizations where the x and y axes need to be inverted.

	Parameters
	----------
	polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
		The input polygon or multipolygon whose coordinates are to be swapped.

	Returns
	-------
	shapely.geometry.Polygon or shapely.geometry.MultiPolygon
		The polygon or multipolygon with swapped x and y coordinates.
	"""

	if polygon.geom_type == 'MultiPolygon':
		
		out_poly = []
		for poly in list(polygon.geoms):
			out_poly.append(transform(lambda x, y: (y, x), poly))
		
		return MultiPolygon(out_poly)
	
	else:   
		return transform(lambda x, y: (y, x), polygon)


def calculate_proximity_density(data, global_con, bool_dict, center_type, peri_type, proxi_dist, alt=False):
	
	"""
	Calculates the density of cells (peri_type) within a specified proximity distance around center cells for downstream neighbor ratio metric calculation, 
	with options for two types of proximity density calculations based on desired neighbor ratio type.

	Parameters
	----------
	data : DataFrame
		The dataset containing cellular information.
	global_con : numpy.ndarray
		A global cellular connectivity matrix representing distances between cells.
	bool_dict : dict
		A dictionary mapping cell types to boolean arrays for indexing the global connectivity matrix.
	center_type : str
		The cell type to be considered as the center cell type of proximity calculations.
	peri_type : str or 'all'
		The peripheral cell type to be considered in the proximity calculation. Use 'all' to consider all cell types.
	proxi_dist : float
		The distance threshold for considering cells to be in proximity (only applies when alt=False, otherwise the thresholds are preset)
	alt : bool, optional
		Determines the method of calculation. False (default) for Type 1 neighbor ratio, True for Type 2.

	Returns
	-------
	list of int
		A list containing the count of cells within proximity and, depending on the mode, either the count of cells
		outside the proximity (30um) or the difference in counts between two proximity thresholds (15um and 30um).

	Notes
	-----
	- The function forms the base for the calculation of the neighbor ratio metrics.
	- It supports two modes of operation: Type 1 neighbor ratio (default) calculates the number of cells within and
	  outside the specified proximity; Type 2 (alt=True) calculates the difference in cell counts within two concentric
	  proximities, offering a more nuanced view of cellular distribution by correcting for large scale architectural differences in tissue architecture
	  and focussing on a smaller neighborhood around the center cell type.
	"""

	#Extract relevant part of global cellular connectivity matrix
	if peri_type == 'all':
		connectivity_matrix = global_con[bool_dict[center_type], : ]
	else:
		connectivity_matrix = global_con[np.ix_(bool_dict[center_type], bool_dict[peri_type])]
	
	#Handle case where number of center cells is 0
	if connectivity_matrix.shape[0] == 0:
		
		if alt == False:
			return [0, connectivity_matrix.shape[1]]
		else:
			return [0, 0]

	#Censor distance to self
	connectivity_matrix[connectivity_matrix == 0] = 10000

	#Type 1 neighbor ratio
	if alt == False:
		
		proxi_graph = connectivity_matrix < proxi_dist
		proxi_num = np.count_nonzero(np.sum(proxi_graph, axis=0))
		non_proxi_num = proxi_graph.shape[1] - proxi_num

		return [proxi_num, non_proxi_num]
	
	#Type 2 neighbor ratio
	if alt == True:
		proxi_graph_inner = connectivity_matrix < (15*3.0769)
		proxi_graph_outer = connectivity_matrix < (30*3.0769)
		proxi_num_inner = np.count_nonzero(np.sum(proxi_graph_inner, axis=0))
		proxi_num_outer = np.count_nonzero(np.sum(proxi_graph_outer, axis=0))
		proxi_diff = proxi_num_outer - proxi_num_inner
		
		return [proxi_num_inner, proxi_diff]


def get_proxi_ratio(data, global_con, bool_dict, center_cells, peri_cells, proxi_dist = 30*um_to_pixel, alt=False): 

	"""
	Systematically applies the calculate_proximity_density() function to all pairs of cell types from the center_cells and peri_cells lists
	and collects results in dictionaries. Used downstream for neighbor ratio metric calculation.

	Parameters
	----------
	data : DataFrame
		The dataset containing cellular information.
	global_con : numpy.ndarray
		A global cellular connectivity matrix representing distances between cells.
	bool_dict : dict
		A dictionary mapping cell types to boolean arrays for indexing the global connectivity matrix.
	center_cells : list of str
		The list of center cell types for which proximity ratios will be calculated.
	peri_cells : list of str
		The list of peripheral cell types to later calculate the neighbor ratios against, for each center cell type.
	proxi_dist : float, optional
		The distance threshold for considering cells to be in proximity (only applies when alt=False, otherwise the thresholds are preset)
	alt : bool, optional
		Determines the method of calculation. False (default) for Type 1 neighbor ratio, True for Type 2.

	Returns
	-------
	center_out_data : dict
		A dictionary containing the results of center cell types to all cells within the proximity distance.
	peri_out_data : dict
		A nested dictionary containing the results of center cell types to each specified peripheral cell
		type within the proximity distance.
	"""

	peri_out_data = {}
	center_out_data = {}
	
	for center_type in center_cells:
		
		center_out_data[center_type] = calculate_proximity_density(data, global_con, bool_dict, center_type, peri_type='all', proxi_dist=proxi_dist, alt=alt)
		peri_out_data[center_type] = {}
		
		for peri_type in peri_cells:
			peri_out_data[center_type][peri_type] = calculate_proximity_density(data, global_con, bool_dict, center_type, peri_type, proxi_dist, alt=alt)

	return center_out_data, peri_out_data


def get_CSR_ratio(data, global_con, bool_dict, center_cells, peri_cells, n_permut = 1000):
	
	"""
	Calculates and compares the spatial relationships between specified center and peripheral cell types within 
	a dataset. It computes the real minimum distances and generates a distribution of minimum distances through 
	permutation testing to assess degree of spatial clustering vs. randomness between pairs of cell types.

	Parameters
	----------
	data : DataFrame
		The dataset containing cellular information.
	global_con : numpy.ndarray
		A global cellular connectivity matrix representing distances between cells.
	bool_dict : dict
		A dictionary mapping cell types to boolean arrays for indexing the global connectivity matrix.
	center_cells : list of str
		The list of center cell types for which CSR ratios will be calculated.
	peri_cells : list of str
		The list of peripheral cell types to later calculate the neighbor ratios against, for each center cell type.
	n_permut : int, optional
		The number of permutations to perform for the permutation test, by default 1000.

	Returns
	-------
	(real_data, permut_data) : tuple of (dict, dict)
		Returns a tuple containing two dictionaries: `real_data` with real minimum distances between each pair 
		of center and peripheral cell types, and `permut_data` with the distribution of minimum distances 
		from the permutation test.

	Notes
	-----
	Not used in final analysis, therefore not called during processing. Included for completeness.
	"""

	real_data = {}
	permut_data = {}
		
	#Iterate through center cells
	for celltype in center_cells:
		
		center_data = data.loc[bool_dict[celltype], ['centroid_x', 'centroid_y']].to_numpy()

		real_data[celltype] = {}
		permut_data[celltype] = {}

		#Handle case of 0 center cells
		if len(center_data) == 0:
			
			for peri in peri_cells:

				real_data[celltype][peri] = np.array([])
				permut_data[celltype][peri] = np.array([])

		else:
			
			permut_medians = {}
			peri_lengths = {}

			#Iterate through peripheral cell types
			for peri_type in peri_cells:
				
				peri_data = data.loc[bool_dict[peri_type], ['centroid_x', 'centroid_y']].to_numpy()
				peri_lengths[peri_type] = len(peri_data)

				#Handle case of 0 peripheral cells
				if peri_lengths[peri_type] == 0:

					real_data[celltype][peri_type] = np.array([])
					permut_data[celltype][peri_type] = np.array([])
					
				else:

					#Extract and save real distances
					connectivity_matrix = global_con[np.ix_(bool_dict[celltype], bool_dict[peri_type])]
					connectivity_matrix[connectivity_matrix == 0] = 10000  #censor distance to self
					CSR_graph = np.min(connectivity_matrix, axis=1)
					real_data[celltype][peri_type] = CSR_graph
					
					permut_medians[peri_type] = np.array([])

					#Run permutations within compartment
					if peri_type.startswith('Tumor'):        
						other_data = global_con[np.ix_(bool_dict[celltype], bool_dict['Tumor'])]
					else:
						other_data = global_con[np.ix_(bool_dict[celltype], ~bool_dict['Tumor'])]
					
					other_data_index = np.arange(other_data.shape[1])
					
					for n in range(n_permut):

						np.random.shuffle(other_data_index)

						#Extract and save shuffled distances
						connectivity_matrix = other_data[:, other_data_index[:peri_lengths[peri_type]]]
						connectivity_matrix[connectivity_matrix == 0] = 10000  #censor distance to self
						CSR_graph = np.min(connectivity_matrix, axis=1)
						permut_medians[peri_type] = np.append(permut_medians[peri_type], CSR_graph)

						# Save data in last iteration
						if n == n_permut - 1:
							permut_data[celltype][peri_type] = permut_medians[peri_type]

	return real_data, permut_data


def calculate_boundary_dist(cell_loc, boundary_lines, Tumor_data, FOV_poly, Mdist):
	
	"""
	Calculates the minimum distance from a cell's location to the nearest tumor boundary while applying censoring to ambiguous cells.

	Parameters
	----------
	cell_loc : Point
		The geometric location of the cell as a Point object.
	boundary_lines : list of LineString
		A list of LineString objects representing various tumor boundary lines within the field of view.
	Tumor_data : list of Polygon
		A list of Polygon objects defining the tumor areas/nests.
	FOV_poly : Polygon
		The Polygon object defining the boundary of the tissue area in the FOV.
	Mdist : float
		The specified distance threshold.

	Returns
	-------
	float
		The minimum distance to the closest tumor boundary for a given cell, with censoring applied.
	"""
	
	Bdists = []
	Tdists = []
	
	#Calculate distance from cell location to the edge of the FOV polygon
	outer_dist = FOV_poly.boundary.distance(cell_loc)
	
	#Calculate distance from cell location to each tumor boundary and tumor area
	for tumor in Tumor_data:
		Tdists.append(tumor.distance(cell_loc))
	for bound in boundary_lines:
		Bdists.append(bound.distance(cell_loc))
		
	#Get the minimum distance or infinity if there is no tumor data
	Tdist = np.min(Tdists) if Tdists else np.inf
	Bdist = np.min(Bdists) if Bdists else np.inf

	#Calculate and return value while applying censoring

	#Legend for returned values:
	# <200   -> cell within boundary area and distance can unambigously be measured
	# 200    -> cell within boundary area but distance cannot unambigously be measured (for example: cells near FOV edge)
	# 1000   -> cell outside of boundary area
	# 10000  -> censored cell (for distance measurements), status cannot be determined unambigously

	if Bdist < Mdist:
			
		if Bdist < outer_dist and Bdist == Tdist:
			return Bdist
		elif Bdist >= outer_dist or Bdist > Tdist:
			return 200
	
	elif Bdist >= Mdist:
		
		if FOV_poly.contains(cell_loc) and outer_dist > Mdist and ((Tdist > Mdist) or (Tdist == 0)):
			return 1000
		else:
			return 10000
	
	

def MeaAs_boundary_dist(data, 
						fig, 
						ax,
						QC_im_name1,
						Tumor_data, 
						granularity=0.006, 
						degree_of_buffering=8*um_to_pixel, 
						Mdist=50*um_to_pixel):
	
	"""
	Analyzes cell data within a field of view (FOV), classifies cells as 'Tumor' or 'Stroma', and calculates their boundary distances.
	This function also generates plots illustrating tumor areas and boundary lines.

	Parameters
	----------
	data : DataFrame
		A DataFrame containing at least 'centroid_x' and 'centroid_y' columns for cell locations, and the 'Origin' column (tumor vs. immune).
	fig, ax : matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
		Figure and axes objects for plotting.
	QC_im_name1 : str
		Path for saving the QC images generated during the analysis.
	Tumor_data : list of Polygon
		A list containing tumor boundary polygons.
	granularity : float, optional
		The granularity for alphashape generation to define the area covered by tissue within the FOV. Default is 0.006.
	degree_of_buffering : int, optional
		Buffer distance for the FOV polygon. Default is 8um.
	Mdist : float, optional
		The margin distance for the tumor boundary area. Default is 50um.

	Returns
	-------
	out_data : DataFrame
		The input DataFrame with added 'Cell_compartment' and 'Boundary_dist' columns indicating cell location / compartment and boundary distances, respectively.
	areas : numpy.ndarray
		An array containing the areas of the whole FOV, as well as total tumor area, buffered inside and outside boundary area.
	"""
	
	#Initialize output
	areas = np.array([0, 0, 0, 0])
	out_data = data.copy()
	
	#Load location data
	loc_matrix_non_tumor = data.loc[data['Origin'] != 'Tumor', ['centroid_x', 'centroid_y']].to_numpy()
	loc_matrix_all = data[['centroid_x', 'centroid_y']].to_numpy()
	
	#Create and buffer the FOV polygon
	FOV_circ = alphashape.alphashape(loc_matrix_all, alpha=granularity)
	FOV_circ = make_valid(FOV_circ)
	FOV_poly = FOV_circ.buffer(distance=degree_of_buffering)
	
	#Calculate the area of the FOV polygon
	areas[0] += FOV_poly.area

	#Ensure tumor areas intersect with POV polygon
	Tumor_data = [tumor.intersection(FOV_poly) for tumor in Tumor_data]
	
	#Classify each cell as 'Tumor' or 'Stroma' based on its location
	loc_bool = np.repeat(False, len(data))
	for tumor in Tumor_data:
		
		if tumor.geom_type == 'MultiPolygon':
			for Tpoly in list(tumor.geoms):
				
				if not Tpoly.is_empty:
					outer_poly_coords = path.Path(Tpoly.boundary.coords)
					contained_points = np.array(outer_poly_coords.contains_points(loc_matrix_all))
					loc_bool = np.vstack([loc_bool, contained_points])
				
		elif tumor.geom_type == 'Polygon' and not tumor.is_empty:           
			outer_poly_coords = path.Path(tumor.boundary.coords)
			contained_points = np.array(outer_poly_coords.contains_points(loc_matrix_all))
			loc_bool = np.vstack([loc_bool, contained_points])

	#Consider all tumor areas simultaneously
	loc_bool = loc_bool.T  
	if loc_bool.ndim > 1:
		loc_bool = np.any(loc_bool, axis=1)
	
	#Shrink FOV by Mdist
	Shrunk_FOV = make_valid(FOV_poly.buffer(distance=-Mdist))

	#For each cell, determine whether it's inside or outside the shrunken FOV polygon
	if Shrunk_FOV.geom_type == 'MultiPolygon':
		FOV_bool = np.repeat(False, len(data))
		for subarea in list(Shrunk_FOV.geoms):
			if not subarea.is_empty:
				outer_poly_coords = path.Path(subarea.boundary.coords)
				contained_points = np.array(outer_poly_coords.contains_points(loc_matrix_all))
				FOV_bool = np.vstack([FOV_bool, contained_points])
			
		FOV_bool = FOV_bool.T  
		if FOV_bool.ndim > 1:
			FOV_bool = np.any(FOV_bool, axis=1)
			
	elif Shrunk_FOV.geom_type == 'Polygon' and not Shrunk_FOV.is_empty:
		outer_poly_coords = path.Path(Shrunk_FOV.boundary.coords)
		FOV_bool = np.array(outer_poly_coords.contains_points(loc_matrix_all))
		
	else:
		FOV_bool = np.repeat(False, len(data))
		
	#Reduce dataset to only include cells either inside the reduced FOV polygon or non-tumor cells in the stroma area
	loc_matrix_shrunk = data.loc[(FOV_bool) | ((~loc_bool) & (data['Origin'] != 'Tumor')), ['centroid_x', 'centroid_y']].to_numpy()
	
	#Create reduced FOV polygon
	corrected_circ = alphashape.alphashape(loc_matrix_shrunk, alpha=granularity)
	corrected_circ = make_valid(corrected_circ)
	corrected_poly = corrected_circ.buffer(distance=degree_of_buffering)
	
	boundary_lines = []
	insides = []
	outsides = []
	
	#Iterate through tumor areas
	for tumor in Tumor_data:

		#Reduce boundary according to reduced FOV polygon
		true_bound = tumor.boundary.intersection(corrected_poly)
		
		#Break up tumor boundary into two pieces if it is closed
		if true_bound.is_closed:

			num_points = len(true_bound.coords)
			break_point_index = int((num_points - (num_points % 2)) / 2)

			part1_coords = true_bound.coords[:break_point_index + 1]
			part2_coords = true_bound.coords[break_point_index:]

			part1_linestring = geometry.LineString(part1_coords)
			part2_linestring = geometry.LineString(part2_coords)
			
			bound_poly1 = make_valid(part1_linestring.buffer(distance=Mdist))
			bound_poly2 = make_valid(part2_linestring.buffer(distance=Mdist))
			
			#Create valid boundary polygon
			bound_poly = make_valid(bound_poly1.union(bound_poly2))
			
		else:

			#Directly create valid boundary polygon
			bound_poly = make_valid(true_bound.buffer(distance=Mdist))
		
		#Extract non-tumor side of boundary area that overlaps with FOV polygon
		check_ext = make_valid(bound_poly.difference(tumor).intersection(FOV_poly))
		
		#Count non-tumor cells within this new polygon
		checksum = 0
		for point in range(loc_matrix_non_tumor.shape[0]):
			P = geometry.Point(loc_matrix_non_tumor[point, : ])
			if P.within(check_ext):
				checksum += 1
			if checksum > 4:
				break
		
		#Require at least 5 non-tumor cells within this polygon to count boundary as valid
		if checksum > 4:

			#Add boundary inside, outside and boundary line
			insides.append(bound_poly.intersection(tumor))
			outsides.append(bound_poly.difference(tumor).intersection(FOV_poly))
			boundary_lines.append(true_bound)
			
			#Plot boundary line
			#Case 1: Single boundary line
			if true_bound.geom_type == 'LineString':
				linepoints = np.array(true_bound.coords)
				scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
				x_coord, y_coord = zip(*scaled_LP)
				ax.plot(y_coord, x_coord, lw=1.5, color='tab:orange')
			
			#Single 2: Fragmented boundary line
			else:
				for geom in true_bound.geoms:
					linepoints = np.array(geom.coords)
					scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
					x_coord, y_coord = zip(*scaled_LP)
					ax.plot(y_coord, x_coord, lw=1.5, color='tab:orange')
		
	#Create final boundary polygons from all tumor areas
	BI_poly = unary_union(insides)
	BO_poly = unary_union(outsides).difference(unary_union(Tumor_data))
	
	#Extract the areas
	areas[1] += sum(tumor.area for tumor in Tumor_data)
	areas[2] += BI_poly.area
	areas[3] += BO_poly.area

	#Update the output dataframe with cell compartments and boundary distances
	out_data['Cell_compartment'] = np.where(loc_bool, 'Tumor', 'Stroma')
	out_data['Boundary_dist'] = [calculate_boundary_dist(geometry.Point(loc), boundary_lines, Tumor_data, FOV_poly, Mdist) for loc in loc_matrix_all]
	  
	#Define the different cellular subsets for QC plotting (censored, within boundary area, ...)
	CO_p = out_data.loc[out_data['Boundary_dist'] == 10000, ['centroid_x', 'centroid_y']].to_numpy()
	CI_p = out_data.loc[out_data['Boundary_dist'] == 200, ['centroid_x', 'centroid_y']].to_numpy()
	NCOT_p = out_data.loc[(out_data['Boundary_dist'] == 1000) & (out_data['Cell_compartment'] == 'Tumor'), ['centroid_x', 'centroid_y']].to_numpy()
	NCONT_p = out_data.loc[(out_data['Boundary_dist'] == 1000) & (out_data['Cell_compartment'] == 'Stroma'), ['centroid_x', 'centroid_y']].to_numpy()
	NCI_p = out_data.loc[out_data['Boundary_dist'] < 200, ['centroid_x', 'centroid_y']].to_numpy()
	
	#Scale the points
	scaled_CO_p = [(x * (1/3), y * (1/3)) for x, y in CO_p]
	scaled_CI_p = [(x * (1/3), y * (1/3)) for x, y in CI_p]
	scaled_NCOT_p = [(x * (1/3), y * (1/3)) for x, y in NCOT_p]
	scaled_NCONT_p = [(x * (1/3), y * (1/3)) for x, y in NCONT_p]
	scaled_NCI_p = [(x * (1/3), y * (1/3)) for x, y in NCI_p]
	
	#Plot the points (with exception handling)
	try:
		ax.scatter(*list(zip(*scaled_CO_p))[::-1], s=5, color='w', alpha=.4, marker='s')
	except Exception as e:
		print(e)
	try:
		ax.scatter(*list(zip(*scaled_CI_p))[::-1], s=6, color='w', marker='x')
	except Exception as e:
		print(e)
	try:
		ax.scatter(*list(zip(*scaled_NCONT_p))[::-1], s=6, color='w', alpha=.7, marker='v')
	except Exception as e:
		print(e)
	try:
		ax.scatter(*list(zip(*scaled_NCOT_p))[::-1], s=5, color='w', alpha=.7)
	except Exception as e:
		print(e)
	try:
		ax.scatter(*list(zip(*scaled_NCI_p))[::-1], s=6, color='w', marker='x')
	except Exception as e:
		print(e)
	
	#Create boundary plot

	#Inside boundary areas
	try:
		if BI_poly.geom_type == 'GeometryCollection':
			for geom in BI_poly.geoms:
				if geom.geom_type == 'LineString':
					linepoints = np.array(geom.coords)
					scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
					x_coord, y_coord = zip(*scaled_LP)
					ax.plot(x_coord, y_coord, lw=1.5, color='b')
				
				else:
					scaled_poly = scale(geom, xfact=1/3, yfact=1/3, origin=(0, 0))
					#patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
					#ax.add_patch(patch) 
		
		else:
			scaled_poly = scale(BI_poly, xfact=1/3, yfact=1/3, origin=(0, 0))
			#patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
			#ax.add_patch(patch) 
		
	except Exception as e:
		print(e, 'b', )
	
	#Outside boundary areas
	try:
		if BO_poly.geom_type == 'GeometryCollection':
			for geom in BO_poly.geoms:
				if geom.geom_type == 'LineString':
					linepoints = np.array(geom.coords)
					scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
					x_coord, y_coord = zip(*scaled_LP)
					ax.plot(x_coord, y_coord, lw=1.5, color='m')
				
				else:
					scaled_poly = scale(geom, xfact=1/3, yfact=1/3, origin=(0, 0))
					#patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
					#ax.add_patch(patch) 
		
		else:
			scaled_poly = scale(BO_poly, xfact=1/3, yfact=1/3, origin=(0, 0))
			#patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
			#ax.add_patch(patch)
		
	except Exception as e:
		print(e, 'c')
		
	#Save QC plot
	plt.savefig(QC_im_name1, transparent=True, dpi=300)
	plt.close('all')
	
	return out_data, areas
	
	
def countDC_clusters(data, min_cluster_size = 5, max_dist = 30*um_to_pixel): 

	"""
	Identifies and counts DC3 clusters within the provided data based on pre-determined criteria.

	Parameters
	----------
	data : DataFrame
		The dataset containing cells with at least columns 'centroid_x', 'centroid_y' for cell locations, and 'identity' to identify cell types.
	min_cluster_size : int, optional
		The minimum number of cells required to count as a cluster. Default is 5.
	max_dist : float, optional
		The maximum distance between cells to consider them part of the same cluster. Default is 30um.

	Returns
	-------
	cluster_num : int
		The number of DC clusters identified within the data.
	out_data : DataFrame
		The input DataFrame updated with a 'DC_cluster_loc' column indicating if a cell is 'Inside' or 'Outside' a DC cluster.
	"""   

	#Initialize outputs
	clusters = []
	cluster_num = 0
	
	#Load array with all cell locations
	loc_matrix_all = data[['centroid_x', 'centroid_y']].to_numpy()
	
	#Generate DC3 clusters
	DC_data = data.loc[data['identity'] == 'DC3', ['unique_ident', 'identity', 'centroid_x', 'centroid_y']].reset_index(drop = False, inplace = False)
	
	loc_matrix = DC_data[['centroid_x', 'centroid_y']].to_numpy()
	connectivity_matrix = scipy.spatial.distance_matrix(loc_matrix, loc_matrix, p=2)
	
	connectivity_matrix[connectivity_matrix < max_dist] = 1
	connectivity_matrix[connectivity_matrix >= max_dist] = 0
	
	#DC3 graph
	total_graph = sparse.csr_matrix(connectivity_matrix)
	
	loc_bool = np.repeat(False, len(data))
	
	#Add a set to store visited nodes
	visited_nodes = set() 

	#Traverse graph from all possible start nodes
	for start_node in range(0, len(loc_matrix)):
		
		#Check if node has been visited before
		if start_node in visited_nodes: 
			continue

		#BFS for completeness (memory usage seems ok)
		conn_graph = sparse.csgraph.breadth_first_order(total_graph, start_node, return_predecessors = False)    
		nodes_list = set(conn_graph)

		#Update visited_nodes set with all nodes in the cluster
		visited_nodes.update(nodes_list)

		#Get all DC3 clusters of sufficient size
		if len(nodes_list) > min_cluster_size:

				#Add to cluster count
				clusters.append(nodes_list)
				cluster_num += 1
				
				#Create cluster alphashape
				points = loc_matrix[conn_graph, : ]
				DC_circ = alphashape.alphashape(points, alpha=0)
				DC_circ = make_valid(DC_circ)
				
				padded_poly = DC_circ.buffer(distance = max_dist)
				outer_poly_coords = path.Path(padded_poly.boundary.coords)
					
				#Check which cells lie within cluster alphashape
				contained_cells = np.array(outer_poly_coords.contains_points(loc_matrix_all))
				loc_bool = np.vstack([loc_bool, contained_cells])

	#Check across all identified clusters
	loc_bool = loc_bool.T  
	if loc_bool.ndim > 1:
		loc_bool = np.any(loc_bool, axis=1)
	
	#Add all cells' locations relative to DC3 cluster (inside/outside) to output dataframe
	out_data = data.copy()
	out_data['DC_cluster_loc'] = np.where(loc_bool, 'Inside', 'Outside')
		   
	return cluster_num, out_data