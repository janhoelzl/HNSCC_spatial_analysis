'''
Functions for part 4 of main HNSCC image analysis pipeline: Spatial processing
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''

##FUNCTIONS

def power_set(items, conds):
    N = len(items)
    out_sets = []
    # enumerate the 2 ** N possible combinations
    for i in range(2 ** N):
        combo = set([])
        for j in range(N):
            # test bit jth of integer i
            if (i >> j) % 2 == 1:
                combo = combo.union(set([items[j]]))
        if conds.issubset(combo):
            out_sets.append(combo)
    return out_sets



def prepare(outdir):
    
    #Create list containing all unique celltype combinations
    
    combs_old = {'Tumor': ['KI67', 'PDL1'],
             'DC3': ['KI67', 'PD1', 'CD163', 'PDL1'], 
             'CD4-Tcell': ['KI67', 'PD1', 'TCF1', 'FoxP3', 'PDL1'], 
             'CD8-Tcell': ['KI67', 'PD1', 'TCF1', 'FoxP3', 'PDL1'], 
             'Bcell': ['KI67', 'PD1', 'PDL1'], 
             'Neutrophil': ['KI67', 'PD1', 'CD163', 'PDL1'], 
             'Macrophage/Monocyte': ['KI67', 'PD1', 'CD163', 'PDL1'],
             'Other-myeloid': ['KI67', 'PD1', 'CD163', 'PDL1'], 
             'Other-CD45': ['KI67', 'PD1', 'PDL1']}
    
    combs = {'Tumor': ['KI67', 'PDL1'],
             'DC3': ['KI67', 'CD163', 'PDL1'], 
             'CD4-Tcell': ['KI67', 'TCF1'], 
             'Treg': ['KI67'],
             'CD8-Tcell': ['KI67', 'TCF1'],
             'Neutrophil': ['KI67','CD163', 'PDL1'], 
             'Macrophage/Monocyte': ['KI67', 'CD163', 'PDL1']}
    
    celltypes_base = ['Tumor',
                     'DC3', 
                     'CD4-Tcell',
                     'Treg',
                     'CD8-Tcell', 
                     'Neutrophil', 
                     'Macrophage/Monocyte']
    
    celltypes = celltypes_base.copy()
    
    #Only one level deep
    for ct in celltypes_base:
        for sub in combs[ct]:
            celltypes.append('_'.join([ct, sub+'pos']))
            celltypes.append('_'.join([ct, sub+'neg']))
      
    #List of base metrics
    base_metrics = ['median_PDL1_expr_total', 'median_PDL1_expr_stroma', 'median_PDL1_expr_tumor', 'median_PDL1_expr_boundary_all', 'median_PDL1_expr_boundary_in', 'median_PDL1_expr_boundary_out',
                   'density_total', 'density_stroma', 'density_tumor', 'density_boundary_all', 'density_boundary_in', 'density_boundary_out',
                   'frequency_total_Comp', 'frequency_stroma_Comp', 'frequency_tumor_Comp',  'frequency_boundary_Comp_all', 'frequency_boundary_Comp_in', 'frequency_boundary_Comp_out',
                   'frequency_total_allCell', 'frequency_stroma_allCell', 'frequency_tumor_allCell', 'frequency_boundary_allCell_all', 'frequency_boundary_allCell_in', 'frequency_boundary_allCell_out']

    
    #Set cross product for final list of base metrics
    metric_list = ['&'.join([i1, i2]) for i1 in base_metrics for i2 in celltypes]
    
    #TLS and DC metrics
    TLS_metrics = ['FullTLS', 'PartialTLS', 'DC_clusters', 'DC_cluster_prop', 'Processed_portion', 'Full_area', 'Tumor_area', 'BI_area', 'BO_area']
    
    #Median boundary dists and other boundary metrics (?)
    Boundary_dist_metrics = ['&'.join([comp, ct]) for comp in ['Tumor_dist', 'Stroma_dist'] for ct in celltypes[1:]]
    
    #Neighborhood analysis
    Celltype_cross_prod = ['X'.join([ct1, ct2]) for ct1 in celltypes for ct2 in celltypes]
    
    #Celltype_cross_prod = Celltype_cross_prod[0:50]
    
    Celltype_dist_metrics = ['&'.join([Measure, pair]) for Measure in ['Neighbor_ratio', 'CSR_ratio'] for pair in Celltype_cross_prod]
    
    #Full metric list
    Full_metrics = [metric_list, TLS_metrics, Boundary_dist_metrics, Celltype_dist_metrics]
    Full_metrics = [item for sublist in Full_metrics for item in sublist]
    print(len(Full_metrics))
    
    with open(f'{outdir}/Metrics_list', 'wb') as file:
        pickle.dump(Full_metrics, file)
    
    return Full_metrics


    def get_TLS_count(data, FOV, Bcell_dist = 30*um_to_pixel, minBcell_cluster_size = 5, degree_of_buffering = 10*um_to_pixel, granularity = 0):    

    part_TLS = 0
    full_TLS = 0
    
    #Generate B cell clusters
    b_cell_data = data.loc[data['identity'] == 'Bcell', ['unique_ident', 'identity', 'centroid_x', 'centroid_y']].reset_index(drop = False, inplace = False)
    
    loc_matrix = b_cell_data[['centroid_x', 'centroid_y']].to_numpy()
    connectivity_matrix = scipy.spatial.distance_matrix(loc_matrix, loc_matrix, p=2)
    
    connectivity_matrix[connectivity_matrix < Bcell_dist] = 1
    connectivity_matrix[connectivity_matrix >= Bcell_dist] = 0
    total_graph = sparse.csr_matrix(connectivity_matrix)
    
    pot_TLS = []
    names = ['FOV', 'type', 'centroid_x', 'centroid_y', 'orig_area', 'padded_area', 'Bcell_cluster_size']
    out_data = pd.DataFrame(columns=names)
    
    visited_nodes = set() # Add a set to store visited nodes

    for start_node in range(0, len(loc_matrix)):
        if start_node in visited_nodes: # Check if this node has been visited before
            continue

        #BFS chosen for completeness (memory seems fine)
        conn_graph = sparse.csgraph.breadth_first_order(total_graph, start_node, return_predecessors = False)    
        nodes_list = set(conn_graph)

        visited_nodes.update(nodes_list) # Update visited_nodes set with all nodes in the cluster
        
        #Get all B cell clusters of sufficient size
        if len(nodes_list) > minBcell_cluster_size:   # and nodes_list not in pot_TLS:

                pot_TLS.append(nodes_list)
                points = loc_matrix[conn_graph, : ]
                TLS_circ = alphashape.alphashape(points, alpha=granularity)
                TLS_circ = make_valid(TLS_circ)
                
                orig_area = TLS_circ.area / (um_to_pixel**2)    #given in um
                padded_poly = TLS_circ.buffer(distance = degree_of_buffering) #join styles default of 1

                centroid_x = TLS_circ.centroid.x
                centroid_y = TLS_circ.centroid.y
                padded_area = padded_poly.area / (um_to_pixel**2)
                cluster_size = len(nodes_list)

                DC_loc = data.loc[data['identity'] == 'DC3', ['centroid_x', 'centroid_y']].to_numpy()
                Tcell_loc = data.loc[data['identity'].str.contains('Tcell'), ['centroid_x', 'centroid_y']].to_numpy()
                    
                outer_poly_coords = path.Path(padded_poly.boundary.coords)
                    
                contained_DC = np.array(outer_poly_coords.contains_points(DC_loc))
                contained_Tcell = np.array(outer_poly_coords.contains_points(Tcell_loc))
                DC_count = contained_DC[contained_DC == True].size
                Tcell_count = contained_Tcell[contained_Tcell == True].size

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
    
    #Prepare out_im filename
    im_name = im_path.split('/')[-1].split('_')
    
    # Load the image into a numpy array
    im = np.array(Img.open(im_path))
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(im, cmap='gray')
    
    #Generate Tumor cell clusters
    cell_data = data.loc[data['identity'] == 'Tumor', ['unique_ident', 'identity', 'centroid_x', 'centroid_y']].reset_index(drop = False, inplace = False)
    
    loc_matrix = cell_data[['centroid_x', 'centroid_y']].to_numpy()
    connectivity_matrix = scipy.spatial.distance_matrix(loc_matrix, loc_matrix, p=2)
    
    connectivity_matrix[connectivity_matrix < cell_dist] = 1
    connectivity_matrix[connectivity_matrix >= cell_dist] = 0
    total_graph = sparse.csr_matrix(connectivity_matrix)
    
    Tumor_nodes = []
    Tumor = []

    visited_nodes = set() # Add a set to store visited nodes
    
    fig2, ax2 = 'a', 'b' #deepcopy((fig, ax))

    for start_node in range(0, len(loc_matrix)):
        if start_node in visited_nodes: # Check if this node has been visited before
            continue

        #BFS chosen for completeness (memory seems fine)
        conn_graph = sparse.csgraph.breadth_first_order(total_graph, start_node, return_predecessors = False)    
        nodes_list = set(conn_graph)

        visited_nodes.update(nodes_list) # Update visited_nodes set with all nodes in the cluster
        
        #Get all Tumor cell clusters of sufficient size
        if len(nodes_list) > min_cluster_size:
                Tumor_nodes.append(nodes_list)
                points = loc_matrix[conn_graph, : ]
                Tumor_circ = alphashape.alphashape(points, alpha=granularity)
                Tumor_circ = make_valid(Tumor_circ)
                padded_poly = Tumor_circ.buffer(distance = degree_of_buffering) #join styles default of 1
                Tumor.append(padded_poly)
                
                # Scale the points and the polygon
                scaled_points = [(x * (1/3), y * (1/3)) for x, y in points]
                scaled_poly = scale(padded_poly, xfact=1/3, yfact=1/3, origin=(0, 0))

                #ax.scatter(*list(zip(*scaled_points))[::-1], s=6, color='tab:orange', alpha=0.2)  # Plot the cell points with flipped coordinates
                patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.22, color='tab:orange')
                ax.add_patch(patch)  # Add the patch to the plot
    
    out_im_name1 = f'{save_dir}/low_scale_circling/{sample}/Tumor_area_{im_name[1]}_{FOV}.png'
    out_im_name2 = f'{save_dir}/low_scale_circling/{sample}/Tumor_bounds_{im_name[1]}_{FOV}_x.png'
    
    return Tumor, Tumor_nodes, fig, ax, fig2, ax2, out_im_name1, out_im_name2



def swap_xy(polygon):
    if polygon.geom_type == 'MultiPolygon':
        out_poly = []
        for poly in list(polygon.geoms):
            out_poly.append(transform(lambda x, y: (y, x), poly))
        return MultiPolygon(out_poly)
    else:   
        return transform(lambda x, y: (y, x), polygon)



def calculate_proximity_density(data, global_con, bool_dict, center_type, peri_type, proxi_dist):
    
    if peri_type == 'all':
        connectivity_matrix = global_con[bool_dict[center_type], : ]
    else:
        connectivity_matrix = global_con[np.ix_(bool_dict[center_type], bool_dict[peri_type])]
    
    if connectivity_matrix.shape[0] == 0:
        return [0, connectivity_matrix.shape[1]]
        #return [0, 0]

    connectivity_matrix[connectivity_matrix == 0] = 10000  #censor distance to self
    proxi_graph = connectivity_matrix < proxi_dist
    proxi_num = np.count_nonzero(np.sum(proxi_graph, axis=0))
    non_proxi_num = proxi_graph.shape[1] - proxi_num
    
    #Alt
    #proxi_graph_inner = connectivity_matrix < proxi_dist #small rad
    #proxi_graph_outer = connectivity_matrix < (60*3.0769) #large rad
    #proxi_num_inner = np.count_nonzero(np.sum(proxi_graph_inner, axis=0))
    #proxi_num_outer = np.count_nonzero(np.sum(proxi_graph_outer, axis=0))
    #proxi_diff = proxi_num_outer - proxi_num_inner
    
    return [proxi_num, non_proxi_num]
    #return [proxi_num_inner, proxi_diff]



def get_proxi_ratio(data, global_con, bool_dict, center_cells, peri_cells, proxi_dist = 30*um_to_pixel): 

    peri_out_data = {}
    center_out_data = {}
    
    for center_type in center_cells:
        center_out_data[center_type] = calculate_proximity_density(data, global_con, bool_dict, center_type, peri_type='all', proxi_dist=proxi_dist)
        peri_out_data[center_type] = {}
        for peri_type in peri_cells:
            peri_out_data[center_type][peri_type] = calculate_proximity_density(data, global_con, bool_dict, center_type, peri_type, proxi_dist)

    return center_out_data, peri_out_data




def get_CSR_ratio(data: pd.DataFrame, global_con: np.array, bool_dict: dict, center_cells: list, peri_cells: list, n_permut: int = 1000) -> tuple:
    
    """
    Calculates the minimum spatial distances between the center cells and peripheral cells 
    and performs a permutation test to shuffle these distances.
    Cave: For now, center_cells and peri_cells are assumed to be equal

    Args:
        data: A DataFrame with columns 'unique_ident', 'identity', 'centroid_x', 'centroid_y'.
        center_cells: A list of center cell types to be analyzed.
        peri_cells: A list of peripheral cell types to be analyzed.
        n_permut: The number of permutations to run.

    Returns:
        A tuple of two dictionaries (real_data and permut_data) which hold the real and permuted 
        minimum distances between each pair of center and peripheral cell types, respectively.
    """

    real_data = {}
    permut_data = {}
        
    for celltype in center_cells:
        
        center_data = data.loc[bool_dict[celltype], ['centroid_x', 'centroid_y']].to_numpy()

        real_data[celltype] = {}
        permut_data[celltype] = {}

        if len(center_data) == 0:
            for peri in peri_cells:
                real_data[celltype][peri] = np.array([])
                permut_data[celltype][peri] = np.array([])

        else:
            
            permut_medians = {}
            peri_lengths = {}

            for peri_type in peri_cells:
                
                peri_data = data.loc[bool_dict[peri_type], ['centroid_x', 'centroid_y']].to_numpy()
                peri_lengths[peri_type] = len(peri_data)

                if peri_lengths[peri_type] == 0:
                    real_data[celltype][peri_type] = np.array([])
                    permut_data[celltype][peri_type] = np.array([])
                    
                else:
                    #loc_matrix_peri = peri_data
                    connectivity_matrix = global_con[np.ix_(bool_dict[celltype], bool_dict[peri_type])]
                    connectivity_matrix[connectivity_matrix == 0] = 10000  #censor distance to self
                    CSR_graph = np.min(connectivity_matrix, axis=1)
                    real_data[celltype][peri_type] = CSR_graph
                    permut_medians[peri_type] = np.array([])

                    # Run permutations within compartment
                    if peri_type.startswith('Tumor'):        
                        other_data = global_con[np.ix_(bool_dict[celltype], bool_dict['Tumor'])]
                    else:
                        other_data = global_con[np.ix_(bool_dict[celltype], ~bool_dict['Tumor'])]
                    
                    other_data_index = np.arange(other_data.shape[1])
                    
                    for n in range(n_permut):
                        np.random.shuffle(other_data_index)
                        connectivity_matrix = other_data[:, other_data_index[:peri_lengths[peri_type]]]
                        connectivity_matrix[connectivity_matrix == 0] = 10000  #censor distance to self
                        CSR_graph = np.min(connectivity_matrix, axis=1)
                        permut_medians[peri_type] = np.append(permut_medians[peri_type], CSR_graph)

                        # Save data in last iteration
                        if n == n_permut - 1:
                            permut_data[celltype][peri_type] = permut_medians[peri_type]

    return real_data, permut_data




def calculate_boundary_dist(cell_loc, boundary_lines, Tumor_data, BI_poly, BO_poly, FOV_poly, Mdist):
    
    """
    Calculate the minimum distance from a given cell location to the tumor boundary and the edge of the field of view (FOV).

    Args:
        cell_loc (Point): The location of the cell.
        Tumor_data (list of Polygon): The tumor data containing tumor boundaries.
        FOV_poly (Polygon): The field of view boundary.
        Mdist (float): The boundary distance.

    Returns:
        float: The minimum boundary distance.
    """
    
    Bdists = []
    Tdists = []
    boundary_union = BI_poly.union(BO_poly)
    
    # Calculate distance from cell location to the edge of the FOV
    outer_dist = FOV_poly.boundary.distance(cell_loc)
    
    # Calculate distance from cell location to each tumor boundary and area
    for tumor in Tumor_data:
        Tdists.append(tumor.distance(cell_loc))
    for bound in boundary_lines:
        Bdists.append(bound.distance(cell_loc))
        
    # Get the minimum distance or infinity if there is no tumor data
    Tdist = np.min(Tdists) if Tdists else np.inf
    Bdist = np.min(Bdists) if Bdists else np.inf

    if Bdist < Mdist: #boundary_union.contains(cell_loc):
            
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
                        fig2,
                        ax2,
                        QC_im_name1,
                        QC_im_name2,
                        Tumor_data, 
                        granularity=0.006, 
                        degree_of_buffering=8*um_to_pixel, 
                        Mdist=50*um_to_pixel):
    
    """
    Calculate the boundary distance for each cell in the given data and classify each cell as 'Tumor' or 'Stroma'.

    Args:
        data (DataFrame): The cell data with 'centroid_x' and 'centroid_y' columns representing cell locations.
        Tumor_data (list of Polygon): The tumor data containing tumor boundaries.
        granularity (float, optional): The granularity for alphashape. Default is 0.005.
        degree_of_buffering (int, optional): The degree of buffering for the FOV polygon. Default is 20.
        Mdist (float, optional): The boundary distance. Default is 100*um_to_pixel.

    Returns:
        DataFrame: The updated cell data with new 'Cell_compartment' and 'Boundary_dist' columns.
        numpy.ndarray: The areas of the whole FOV and tumor areas
    """
    
    areas = np.array([0, 0, 0, 0])
    out_data = data.copy()
    loc_matrix_non_tumor = data.loc[data['Origin'] != 'Tumor', ['centroid_x', 'centroid_y']].to_numpy()
    loc_matrix_all = data[['centroid_x', 'centroid_y']].to_numpy()
    
    # Create and buffer the field of view polygon
    FOV_circ = alphashape.alphashape(loc_matrix_all, alpha=granularity)
    FOV_circ = make_valid(FOV_circ)
    FOV_poly = FOV_circ.buffer(distance=degree_of_buffering)
    # Calculate the area of the field of view
    areas[0] += FOV_poly.area

    Tumor_data = [tumor.intersection(FOV_poly) for tumor in Tumor_data]
    
    # Classify each cell as 'Tumor' or 'Stroma' based on its location
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

    loc_bool = loc_bool.T  
    if loc_bool.ndim > 1:
        loc_bool = np.any(loc_bool, axis=1)
    
    #Shrink FOV
    Shrunk_FOV = make_valid(FOV_poly.buffer(distance=-Mdist))

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
        
    loc_matrix_shrunk = data.loc[(FOV_bool) | ((~loc_bool) & (data['Origin'] != 'Tumor')), ['centroid_x', 'centroid_y']].to_numpy()
    
    corrected_circ = alphashape.alphashape(loc_matrix_shrunk, alpha=granularity)
    corrected_circ = make_valid(corrected_circ)
    corrected_poly = corrected_circ.buffer(distance=degree_of_buffering)
    
    boundary_lines = []
    insides = []
    outsides = []
    
    for tumor in Tumor_data:
        #Create list of true tumor boundaries
        true_bound = tumor.boundary.intersection(corrected_poly)
        
        if true_bound.is_closed:
            num_points = len(true_bound.coords)
            break_point_index = int((num_points - (num_points % 2)) / 2)

            part1_coords = true_bound.coords[:break_point_index + 1]
            part2_coords = true_bound.coords[break_point_index:]

            part1_linestring = geometry.LineString(part1_coords)
            part2_linestring = geometry.LineString(part2_coords)
            
            bound_poly1 = make_valid(part1_linestring.buffer(distance=Mdist))
            bound_poly2 = make_valid(part2_linestring.buffer(distance=Mdist))
            bound_poly = make_valid(bound_poly1.union(bound_poly2))
            
        else:
            bound_poly = make_valid(true_bound.buffer(distance=Mdist))
            
        check_ext = make_valid(bound_poly.difference(tumor).intersection(FOV_poly))
        
        checksum = 0
        for point in range(loc_matrix_non_tumor.shape[0]):
            P = geometry.Point(loc_matrix_non_tumor[point, : ])
            if P.within(check_ext):
                checksum += 1
            if checksum > 4:
                break
                
        if checksum > 4:
            insides.append(bound_poly.intersection(tumor))
            outsides.append(bound_poly.difference(tumor).intersection(FOV_poly))
            boundary_lines.append(true_bound)
            
            #Plot boundary line
            if true_bound.geom_type == 'LineString':
                linepoints = np.array(true_bound.coords)
                scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
                x_coord, y_coord = zip(*scaled_LP)
                ax.plot(y_coord, x_coord, lw=1.5, color='tab:orange')
                
            else:
                for geom in true_bound.geoms:
                    linepoints = np.array(geom.coords)
                    scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
                    x_coord, y_coord = zip(*scaled_LP)
                    ax.plot(y_coord, x_coord, lw=1.5, color='tab:orange')
        
    #Create final boundary polygons
    BI_poly = unary_union(insides)
    BO_poly = unary_union(outsides).difference(unary_union(Tumor_data))
    
    # Calculate the areas
    areas[1] += sum(tumor.area for tumor in Tumor_data)
    areas[2] += BI_poly.area
    areas[3] += BO_poly.area

    # Update the cell data with the cell compartments and the boundary distances
    out_data['Cell_compartment'] = np.where(loc_bool, 'Tumor', 'Stroma')
    out_data['Boundary_dist'] = [calculate_boundary_dist(geometry.Point(loc), boundary_lines, Tumor_data, BI_poly, BO_poly, FOV_poly, Mdist) for loc in loc_matrix_all]
      
    #Define the different point-sets
    CO_p = out_data.loc[out_data['Boundary_dist'] == 10000, ['centroid_x', 'centroid_y']].to_numpy()
    CI_p = out_data.loc[out_data['Boundary_dist'] == 200, ['centroid_x', 'centroid_y']].to_numpy()
    NCOT_p = out_data.loc[(out_data['Boundary_dist'] == 1000) & (out_data['Cell_compartment'] == 'Tumor'), ['centroid_x', 'centroid_y']].to_numpy()
    NCONT_p = out_data.loc[(out_data['Boundary_dist'] == 1000) & (out_data['Cell_compartment'] == 'Stroma'), ['centroid_x', 'centroid_y']].to_numpy()
    NCI_p = out_data.loc[out_data['Boundary_dist'] < 200, ['centroid_x', 'centroid_y']].to_numpy()
    
     # Scale and plot the points
    scaled_CO_p = [(x * (1/3), y * (1/3)) for x, y in CO_p]
    scaled_CI_p = [(x * (1/3), y * (1/3)) for x, y in CI_p]
    scaled_NCOT_p = [(x * (1/3), y * (1/3)) for x, y in NCOT_p]
    scaled_NCONT_p = [(x * (1/3), y * (1/3)) for x, y in NCONT_p]
    scaled_NCI_p = [(x * (1/3), y * (1/3)) for x, y in NCI_p]
    
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


    #Create and and save plots
    #First: Tumor area plot
    '''
    try:
        scaled_poly = scale(FOV_poly, xfact=1/3, yfact=1/3, origin=(0, 0))
        patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.15, color='g')
        ax.add_patch(patch)  # Add the patch to the plot
    
    except Exception as e:
        print(e, 'a')
        
    plt.savefig(QC_im_name1, transparent=True, dpi=300)
    ax.patches.pop()'''
    
    #Now: Boundary plot
    try:
        if BI_poly.geom_type == 'GeometryCollection':
            for geom in BI_poly.geoms:
                if geom.geom_type == 'LineString':
                    linepoints = np.array(geom.coords)
                    scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
                    x_coord, y_coord = zip(*scaled_LP)
                    #x_coord = x_coord[::-1]
                    #y_coord = y_coord[::-1]
                    ax.plot(x_coord, y_coord, lw=1.5, color='b')
                
                else:
                    scaled_poly = scale(geom, xfact=1/3, yfact=1/3, origin=(0, 0))
                    patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
                    ax.add_patch(patch) 
        else:
            scaled_poly = scale(BI_poly, xfact=1/3, yfact=1/3, origin=(0, 0))
            patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
            ax.add_patch(patch) 
        
    except Exception as e:
        print(e, 'b', )
        
    try:
        if BO_poly.geom_type == 'GeometryCollection':
            for geom in BO_poly.geoms:
                if geom.geom_type == 'LineString':
                    linepoints = np.array(geom.coords)
                    scaled_LP = np.array([(x * (1/3), y * (1/3)) for x, y in linepoints])
                    x_coord, y_coord = zip(*scaled_LP)
                    #x_coord = x_coord[::-1]
                    #y_coord = y_coord[::-1]
                    ax.plot(x_coord, y_coord, lw=1.5, color='m')
                
                else:
                    scaled_poly = scale(geom, xfact=1/3, yfact=1/3, origin=(0, 0))
                    patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
                    ax.add_patch(patch) 
        else:
            scaled_poly = scale(BO_poly, xfact=1/3, yfact=1/3, origin=(0, 0))
            patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.25, color='w')
            ax.add_patch(patch)
        
    except Exception as e:
        print(e, 'c')
        
    #scaled_poly = scale(corrected_poly, xfact=1/3, yfact=1/3, origin=(0, 0))
    #patch = PolygonPatch(swap_xy(scaled_poly), alpha=0.15, color='g')
    #ax.add_patch(patch) 
    plt.savefig(QC_im_name2, transparent=True, dpi=300)
  
    plt.close('all')
    
    return out_data, areas
    
    
def countDC_clusters(data, min_cluster_size = 5, max_dist = 30*um_to_pixel):    

    clusters = []
    cluster_num = 0
    
    loc_matrix_all = data[['centroid_x', 'centroid_y']].to_numpy()
    
    #Generate B cell clusters
    DC_data = data.loc[data['identity'] == 'DC3', ['unique_ident', 'identity', 'centroid_x', 'centroid_y']].reset_index(drop = False, inplace = False)
    
    loc_matrix = DC_data[['centroid_x', 'centroid_y']].to_numpy()
    connectivity_matrix = scipy.spatial.distance_matrix(loc_matrix, loc_matrix, p=2)
    
    #Connectivity matrix
    connectivity_matrix[connectivity_matrix < max_dist] = 1
    connectivity_matrix[connectivity_matrix >= max_dist] = 0
    total_graph = sparse.csr_matrix(connectivity_matrix)
    
    visited_nodes = set() # Add a set to store visited nodes
    loc_bool = np.repeat(False, len(data))
    
    for start_node in range(0, len(loc_matrix)):
        if start_node in visited_nodes: # Check if this node has been visited before
            continue

        #BFS chosen for completeness (memory seems fine)
        conn_graph = sparse.csgraph.breadth_first_order(total_graph, start_node, return_predecessors = False)    
        nodes_list = set(conn_graph)

        visited_nodes.update(nodes_list)
        
        #Get all B cell clusters of sufficient size
        if len(nodes_list) > min_cluster_size:   # and nodes_list not in clusters:

                clusters.append(nodes_list)
                cluster_num += 1
                
                points = loc_matrix[conn_graph, : ]
                DC_circ = alphashape.alphashape(points, alpha=0)
                DC_circ = make_valid(DC_circ)
                
                padded_poly = DC_circ.buffer(distance = max_dist) #join styles default of 1
                outer_poly_coords = path.Path(padded_poly.boundary.coords)
                    
                contained_cells = np.array(outer_poly_coords.contains_points(loc_matrix_all))
                loc_bool = np.vstack([loc_bool, contained_cells])

    loc_bool = loc_bool.T  
    if loc_bool.ndim > 1:
        loc_bool = np.any(loc_bool, axis=1)
        
    out_data = data.copy()
    out_data['DC_cluster_loc'] = np.where(loc_bool, 'Inside', 'Outside')
           

    return cluster_num, out_data