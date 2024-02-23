'''
Processing code for part 4 of main HNSCC image analysis pipeline: Spatial processing
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''

##IMPORTS

import os
import sys
import glob
import math
import threading
import functools
import time
import pickle
import shutil
import alphashape
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import path

import scipy

import skimage
from skimage.color import label2rgb
from skimage.io import imread
from skimage import draw
from PIL import ImageTk, ImageEnhance
from PIL import Image as Img

import cv2

import sklearn as skl
import statistics as stat
from scipy import sparse

from shapely import geometry
from shapely.ops import transform
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from shapely.affinity import scale

from descartes import PolygonPatch
from sklearn.neighbors import KernelDensity

from copy import deepcopy



##VARIABLES TO SET

um_to_pixel = 3.0769

sample_list = ['PIO2']

joined_samples = {'PIO47': [0, 1], 'PIO53': [0, 1], 'PIO10': [0, 1], 'PIO1': [0, 1], 
                  'PIO12': [0, 1], 'PIO16': [0, 1], 'PIO26': [0, 1], 'PIO2': [0, 1], 
                  'PIO6': [0, 1], 'PIO23': [0, 1], 'PIO27': [0, 1], 'PIO15': [0, 1], 
                  'PIO30': [2, 1, 2], 'PIO35': [0, 1], 'PIO24': [0, 1], 'PIO29': [0, 1],
                  'PIO8': [0, 1], 'PIO41': [0, 1], 'PIO52': [0, 1], 'PIO11': [1, 1, 2],
                  'PIO19': [2, 1, 2], 'PIO22': [0, 1], 'PIO28': [0, 1], 'PIO45': [1, 1, 2],
                  'PIO8': [0, 1], 'PIO9': [2, 1, 2], 'PIO38': [0, 1], 'PIO3': [0, 1],
                  'PIO4': [0, 1], 'PIO46': [1, 1, 2], 'PIO48': [0, 1], 'PIO49': [0, 1],
                  'PIO17': [0, 1], 'PIO25': [0, 1], 'PIO50': [0, 1], 'PIO51': [0, 1],
                  'PIO37': [0, 1], 'PIO34': [1, 1, 2], 'PIO13': [1, 1, 2], 'PIO18':[0, 1],
                  'PIO39': [2, 1, 2], 'PIO31': [0, 1]}

#List entries are coded: Pos1: 0 - simple case, 1 - precombined case, 2 - to be combined
#        if 1|2          Pos2 - Pos(-1): numbers of imaged areas

im_log_file = pd.read_csv('path_of_choice')
im_log_file.index = im_log_file['Sample']

FOV_drop = {sample: im_log_file.loc[sample, 'Drop'].split(',') for sample in sample_list}

#Use for all measurements
intensity_metric = 'IQmean_0.05-0.95'



##PROCESSING FUNCTIONS

def gather_all_metrics(sample_list, joined_samples, QC_im_dir='abc', qsub_im_dir='abc', metrics_list='abc'):
    
    #Load list of metrics
    with open(metrics_list, 'rb') as file:
        Full_metrics = pickle.load(file)

    all_data = pd.DataFrame(data = np.zeros((len(sample_list), len(Full_metrics))), index = sample_list, columns = Full_metrics)
                            
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
    
    #Iterate through samples and collect data 
    
    #Imaging metadata
    im_meta = pd.read_csv('path_of_choice')
    failed_FOV = {}
    
    for sample in sample_list:
        
        #Load and prepare data
        if joined_samples[sample][0] == 0:
            sample_data = pd.read_csv(f'path_of_choice/{sample}/Phenotype_data_subset.csv')
            PDL1_data = pd.read_csv(f'path_of_choice/{sample}/Phenotype_data_PDL1.csv')
            sample_data['PDL1-status'] = PDL1_data['PDL1-status']
            
            raw_data = pd.read_csv(f'path_of_choice/{sample}/Data.csv')
        
        
        elif joined_samples[sample][0] == 1:
            sample_data = pd.read_csv(f'path_of_choice/{sample}_comb/Phenotype_data_subset.csv')
            PDL1_data = pd.read_csv(f'path_of_choice/{sample}_comb/Phenotype_data_PDL1.csv')
            sample_data['PDL1-status'] = PDL1_data['PDL1-status']
            
            raw_data = pd.read_csv(f'path_of_choice/{sample}_{joined_samples[sample][1]}/Data.csv')
            
            for subsample in joined_samples[sample][2:]:
                add_raw = pd.read_csv(f'path_of_choice/{sample}_{str(subsample)}/Data.csv')
                
                replacement = '_t'+str(subsample)
                
                add_raw['FOV'] = add_raw['FOV'].str.replace('_t1', replacement)
                add_raw['unique_ident'] = add_raw['unique_ident'].str.replace('_t1', replacement)
                raw_data = raw_data.append(add_raw)
                

        elif joined_samples[sample][0] == 2:
            sample_data = pd.read_csv(f'path_of_choice/{sample}_{joined_samples[sample][1]}/Phenotype_data_subset.csv')
            PDL1_data = pd.read_csv(f'path_of_choice/{sample}_{joined_samples[sample][1]}/Phenotype_data_PDL1.csv')
            sample_data['PDL1-status'] = PDL1_data['PDL1-status']
            
            raw_data = pd.read_csv(f'path_of_choice/{sample}_{joined_samples[sample][1]}/Data.csv')
            
            for subsample in joined_samples[sample][2:]:
                add_sample = pd.read_csv(f'path_of_choice/{sample}_{str(subsample)}/Phenotype_data_subset.csv')
                add_PDL1 = pd.read_csv(f'path_of_choice/{sample}_{str(subsample)}/Phenotype_data_PDL1.csv')
                add_sample['PDL1-status'] = add_PDL1['PDL1-status']
                
                add_raw = pd.read_csv(f'path_of_choice/{sample}_{str(subsample)}/Data.csv')
                
                replacement = '_t'+str(subsample)
                
                add_sample['FOV'] = add_sample['FOV'].str.replace('_t1', replacement)
                add_sample['unique_ident'] = add_sample['unique_ident'].str.replace('_t1', replacement)
                sample_data = sample_data.append(add_sample)
                
                add_raw['FOV'] = add_raw['FOV'].str.replace('_t1', replacement)
                add_raw['unique_ident'] = add_raw['unique_ident'].str.replace('_t1', replacement)
                raw_data = raw_data.append(add_raw)
                
                
                
        #Correct KI67 column
        sample_data['KI67-status'] = ['KI67']*len(sample_data) + sample_data['KI67-status']
        
        #Correct cell types
        macrophage_filter = sample_data['identity'] == 'Macrophage'
        monocyte_filter = sample_data['identity'] == 'Monocyte'
        CD4_filter = sample_data['identity'] == 'CD4-Tcell'
        FoxP3_filter = sample_data['FoxP3-status'] == 'FoxP3pos'
            
        sample_data.loc[macrophage_filter | monocyte_filter, 'identity'] = 'Macrophage/Monocyte'
        sample_data.loc[CD4_filter & FoxP3_filter, 'identity'] = 'Treg'
        
        if os.path.isdir(f'{QC_im_dir}/low_scale_circling/{sample}') == False:
            os.makedirs(f'{QC_im_dir}/low_scale_circling/{sample}')
        
        #red_sample = sample.split('_')[0]
        #Tumor_FOV = np.unique(sample_data['FOV']) 
        
        #Single FOV data dict
        single_FOV_in = {}

        if joined_samples[sample][0] != 0:
        
            subsamples = {str(i): '_'.join([sample, str(i)]) for i in joined_samples[sample][1:]}
            Tumor_FOV = []
            
            for s, curr_sample in subsamples.items():
            
                if len(list(filter(lambda x: curr_sample in x and 'AllTumor' in x, os.listdir('path_of_choice/tumor_circling')))) > 0:
                    Tumor_FOV = [*Tumor_FOV, *list(filter(lambda x: '_t'+s in x, np.unique(sample_data['FOV'])))]
                    
                else:

                    #Filter out non-tumor (by circling) in subsample A
                    NRowCol = NRowCol = [int(im_meta.loc[im_meta['ID'] == curr_sample, '#row'].to_list()[0]), int(im_meta.loc[im_meta['ID'] == curr_sample, '#col'].to_list()[0])]

                    circle_img_path = list(filter(lambda x: curr_sample in x and 'TumorCircleFilled_RGB' in x, os.listdir('path_of_choice/tumor_circling')))
                    circle_img = imread('path_of_choice/tumor_circling/'+circle_img_path[0])

                    # Create an empty array with the same shape as your image
                    circle_arr = np.zeros(circle_img.shape, dtype=np.uint8)

                    # Define the size of each rectangle in the grid
                    rect_height = circle_img.shape[0] / NRowCol[0]
                    rect_width = circle_img.shape[1] / NRowCol[1]

                    circle_grid = np.zeros(NRowCol)
                    circle_arr[np.where((circle_img[:, :, 0] == 255) & (circle_img[:, :, 1] == 255) & (circle_img[:, :, 2] == 0))] = 1

                    good_FOV = []
                    # Check the overlap between each rectangle and the hand-drawn circle
                    for i in range(NRowCol[0]):
                        for j in range(NRowCol[1]):
                            # Define the boundaries of the current rectangle
                            x_min, y_min = int(j * rect_width), int(i * rect_height)
                            x_max, y_max = int((j + 1) * rect_width), int((i + 1) * rect_height)

                            # Calculate the degree of overlap between the current rectangle and the hand-drawn circle
                            rect_points = circle_arr[y_min:y_max+1, x_min:x_max+1]
                            circle_grid[i, j] = np.mean(rect_points)
                            
                            single_FOV_in[f's{str(j*NRowCol[0]+i+1)}_t{s}'] = np.mean(rect_points)

                            if circle_grid[i, j] > 0:
                                good_FOV.append(j*NRowCol[0]+i+1) 
                                
                    if os.path.isdir(f'{QC_im_dir}/large_scale_circling/{curr_sample}') == False:
                        os.makedirs(f'{QC_im_dir}/large_scale_circling/{curr_sample}')

                    # Visualize the grid
                    print(f'Tumor circling results for {curr_sample}.')
                    plt.imshow(circle_grid, cmap='gray')
                    #plt.savefig(f'{QC_im_dir}/large_scale_circling/{curr_sample}/Tumor_circling_{curr_sample}.png', transparent=True)
                    plt.show()
                    plt.close('all')

                    Tumor_FOV = [*Tumor_FOV, *[f's{num}_t{s}' for num in good_FOV]]

        else:
            
            if len(list(filter(lambda x: sample+'_' in x and 'AllTumor' in x, os.listdir('path_of_choice/tumor_circling')))) > 0:
                Tumor_FOV = np.unique(sample_data['FOV'])
                print('All tumor.')
                
            else:
                #Filter out non-tumor (by circling) in whole sample
                NRowCol = [int(im_meta.loc[im_meta['ID'] == sample, '#row'].to_list()[0]), int(im_meta.loc[im_meta['ID'] == sample, '#col'].to_list()[0])]

                circle_img_path = list(filter(lambda x: sample+'_' in x and 'TumorCircleFilled_RGB' in x, os.listdir('path_of_choice/tumor_circling')))
                circle_img = imread('path_of_choice/tumor_circling/'+circle_img_path[0])

                # Create an empty array with the same shape as your image
                circle_arr = np.zeros(circle_img.shape, dtype=np.uint8)

                # Define the size of each rectangle in the grid
                rect_height = circle_img.shape[0] / NRowCol[0]
                rect_width = circle_img.shape[1] / NRowCol[1]

                circle_grid = np.zeros(NRowCol)
                circle_arr[np.where((circle_img[:, :, 0] == 255) & (circle_img[:, :, 1] == 255) & (circle_img[:, :, 2] == 0))] = 1

                good_FOV = []
                # Check the overlap between each rectangle and the hand-drawn circle
                for i in range(NRowCol[0]):
                    for j in range(NRowCol[1]):
                        # Define the boundaries of the current rectangle
                        x_min, y_min = int(j * rect_width), int(i * rect_height)
                        x_max, y_max = int((j + 1) * rect_width), int((i + 1) * rect_height)

                        # Calculate the degree of overlap between the current rectangle and the hand-drawn circle
                        rect_points = circle_arr[y_min:y_max+1, x_min:x_max+1]
                        circle_grid[i, j] = np.mean(rect_points)
                        
                        single_FOV_in[f's{str(j*NRowCol[0]+i+1)}_t1'] = np.mean(rect_points)
                        
                        if circle_grid[i, j] > 0:
                            good_FOV.append(j*NRowCol[0]+i+1) 
                            
                if os.path.isdir(f'{QC_im_dir}/large_scale_circling/{sample}') == False:
                    os.makedirs(f'{QC_im_dir}/large_scale_circling/{sample}')

                # Visualize the grid
                print(f'Tumor circling results for {sample}.')
                plt.imshow(circle_grid, cmap='gray')
                #plt.savefig(f'{QC_im_dir}/large_scale_circling/{sample}/Tumor_circling_{sample}.png', transparent=True)
                plt.show()
                plt.close('all')

                Tumor_FOV = [f's{num}_t1' for num in good_FOV]
                
        print(Tumor_FOV)
        #Filter tumor FOV
        use_FOV = []
        for FOV in Tumor_FOV:
            FOV_data = sample_data.loc[sample_data['FOV'] == FOV]
            
            #Filter 'bad' FOVs and those with < 25 cells
            if not FOV in FOV_drop[sample] and len(FOV_data) > 25:
                use_FOV.append(FOV)
        
        Tumor_FOV = use_FOV
        
        single_FOV_df = pd.DataFrame(single_FOV_in, index=[0])
        print(single_FOV_df.shape)
        single_FOV_df.to_csv(f'path_of_choice/Single_FOV_TumorProp_{sample}.csv')
        single_FOV_in = {}
                
        print(Tumor_FOV)
        #Reduce sample dataframe to tumor FOVs.  
        red_sample_data = sample_data.loc[sample_data['FOV'].str.contains('|'.join(Tumor_FOV)), ]
                                             
        #Assign compartments, measure boundary dists and generate FOV wise metrics
        out_data, all_areas, DC_clusters, Neigh_ratio_dict, proc_FOV, fail_FOV, single_FOV_out, Full_TLS, Partial_TLS = get_add_metrics(sample, joined_samples, red_sample_data, QC_im_dir, qsub_im_dir, celltypes, celltypes, single_FOV_in)
        red_sample_data = out_data.copy()
        
        failed_FOV[sample] = fail_FOV
        
        #global intermed_data 
        #intermed_data = [out_data, all_areas, Full_TLS, Partial_TLS, DC_clusters, Neigh_ratio_dict, CSR_ratio_dict]
        
        #Add raw PD-L1 data
        corr_PDL1 = list(raw_data.loc[raw_data['unique_ident'].isin(red_sample_data['unique_ident']), '_'.join([intensity_metric, 'cycle1_w2'])])
        red_sample_data['_'.join([intensity_metric, 'cycle1_w2'])] = corr_PDL1

        #Saving data table
        #red_sample_data.to_csv(f'/Users/janholzl/Dropbox (Partners HealthCare)/Better_CPS_analysis/Phenotyping_final/data_processed/downstream_processing_and_ML_final/Phenotype_data_augmented_Radius30_{sample}.csv')
        single_FOV_df2 = pd.DataFrame(single_FOV_out)
        #single_FOV_df2.to_csv(f'/Users/janholzl/Dropbox (Partners HealthCare)/Better_CPS_analysis/Phenotyping_final/data_processed/downstream_processing_and_ML_final/Single_FOV_NeighborRatio30_{sample}.csv')
        
        #areas and DC clusters (+FOV QC)
        if len(red_sample_data[(red_sample_data['identity'] == 'DC3')]) != 0:
            all_data.loc[sample, 'DC_cluster_prop'] = len(red_sample_data[(red_sample_data['DC_cluster_loc'] == 'Inside') & (red_sample_data['identity'] == 'DC3')]) / len(red_sample_data[(red_sample_data['identity'] == 'DC3')])
        else:
            all_data.loc[sample, 'DC_cluster_prop'] = np.nan
         
        all_data.loc[sample, 'DC_clusters'] = DC_clusters
        all_data.loc[sample, 'Processed_portion'] = proc_FOV
        all_data.loc[sample, 'Full_area'] = all_areas[0]
        all_data.loc[sample, 'Tumor_area'] = all_areas[1]
        all_data.loc[sample, 'BI_area'] = all_areas[2]
        all_data.loc[sample, 'BO_area'] = all_areas[3]
        all_data.loc[sample, 'FullTLS'] = Full_TLS
        all_data.loc[sample, 'PartialTLS'] = Partial_TLS
        
        CP_dict = {}
        for ctA in celltypes:
            for ctB in celltypes:
                ct = 'X'.join([ctA, ctB])
                CP_dict['&'.join(['Neighbor_ratio', ct])] = Neigh_ratio_dict[ctA][ctB]
            
        all_data.loc[sample, CP_dict.keys()] = pd.Series(CP_dict)
        
        
        for ct in celltypes: 

            #Create tempID column containing matchable ID  
            ct_list = ct.split('_')
            props_list = ['-'.join([prop[:-3], 'status']) for prop in ct_list[1:]]
            red_sample_data['tempID'] = red_sample_data[['identity', *props_list]].apply(lambda x: '_'.join(x.dropna().astype(str)), axis=1)
            
            ct_filter = red_sample_data['tempID'] == ct
            tumor_filter = (red_sample_data['Cell_compartment'] == 'Tumor') & ct_filter
            stroma_filter = (red_sample_data['Cell_compartment'] == 'Stroma') & ct_filter
            boundary_filter = red_sample_data['Boundary_dist'] <= 200
            strict_boundary_filter = red_sample_data['Boundary_dist'] < 200
            prolif_filter = red_sample_data['KI67-status'] == 'KI67pos'

            #For frequencies and densities return 0 in case no such cells present, np.nan if no area
            all_data.loc[sample, '&'.join(['density_total', ct])] = ct_filter.sum()
            all_data.loc[sample, '&'.join(['density_stroma', ct])] = stroma_filter.sum()
            all_data.loc[sample, '&'.join(['density_tumor', ct])] = tumor_filter.sum()
            all_data.loc[sample, '&'.join(['density_boundary_all', ct])] = ((boundary_filter) & (ct_filter)).sum()
            all_data.loc[sample, '&'.join(['density_boundary_in', ct])] = ((boundary_filter) & (tumor_filter)).sum()
            all_data.loc[sample, '&'.join(['density_boundary_out', ct])] = ((boundary_filter) & (stroma_filter)).sum()
            
            all_data.loc[sample, '&'.join(['frequency_total_Comp', ct])] = (ct_filter & prolif_filter).sum() / ct_filter.sum()
            all_data.loc[sample, '&'.join(['frequency_stroma_Comp', ct])] = (stroma_filter & prolif_filter).sum() / stroma_filter.sum()
            all_data.loc[sample, '&'.join(['frequency_tumor_Comp', ct])] = (tumor_filter & prolif_filter).sum() / tumor_filter.sum()
            all_data.loc[sample, '&'.join(['frequency_boundary_Comp_all', ct])] = ((boundary_filter) & (ct_filter) & (prolif_filter)).sum() / ((boundary_filter) & (ct_filter)).sum()
            all_data.loc[sample, '&'.join(['frequency_boundary_Comp_in', ct])] = ((boundary_filter) & (tumor_filter) & (prolif_filter)).sum() / ((boundary_filter) & (tumor_filter)).sum()
            all_data.loc[sample, '&'.join(['frequency_boundary_Comp_out', ct])] = ((boundary_filter) & (stroma_filter) & (prolif_filter)).sum() / ((boundary_filter) & (stroma_filter)).sum()

        #all_data.to_csv(f'/Users/janholzl/Dropbox (Partners HealthCare)/Better_CPS_analysis/Phenotyping_final/data_processed/downstream_processing_and_ML_final/Intermediary_data_Radius30_{sample}.csv')
    
    return all_data, failed_FOV              
                            
           
                            

def get_add_metrics(sample, joined_samples, sample_data, save_dir, qsub_im_dir, center_cells, peri_cells, single_FOV_dict):
    
    'Calculates additional metrics after phenotyping.'

    FOV_list = list(sample_data['FOV'].unique())

    Full_TLS = 0
    Partial_TLS = 0
    DC_clust = 0
    
    names = ['FOV', 'type', 'centroid_x', 'centroid_y', 'orig_area', 'padded_area', 'Bcell_cluster_size']
    TLS_data = pd.DataFrame(columns=names)

    Tumor_polys = {}
    all_real_CSR = {}
    all_permut_CSR = {}
    all_center_data = {}
    all_peri_data = {}
                            
    all_areas = np.array([0, 0, 0, 0])
    
    c = 0
    fail_FOV = []
    
    #Process each FOV individually
    for FOV in FOV_list:
        
        FOV_data = sample_data.loc[sample_data['FOV'] == FOV]

        try:
             
            red_FOV = FOV.split('_')[0]

            if joined_samples[sample][0] == 0:
                aug_sample = sample
            else:
                aug_sample = '_'.join([sample, FOV.split('_')[1][1]])
                print(aug_sample)

            bool_dict = {}
            #Fill boolean dictionary
            for celltype in center_cells:
                ct_list = celltype.split('_')
                props_list = ['-'.join([prop[:-3], 'status']) for prop in ct_list[1:]]
                id_arr = FOV_data[['identity', *props_list]].apply(lambda x: '_'.join(x.dropna().astype(str)), axis=1)
                bool_arr = id_arr == celltype
                bool_dict[celltype] = np.array(bool_arr)

            #Create global connectivity matrix
            loc_matrix_global = FOV_data[['centroid_x', 'centroid_y']].to_numpy()
            global_con = scipy.spatial.distance_matrix(loc_matrix_global, loc_matrix_global, p=2)
            
            #TLS
            fF, pF, dF = get_TLS_count(FOV_data, FOV,
                                       Bcell_dist = 30*um_to_pixel,
                                       minBcell_cluster_size = 4,
                                       degree_of_buffering = 30*um_to_pixel, 
                                       granularity = 0)

            Full_TLS += fF
            Partial_TLS += pF
            TLS_data = pd.concat([TLS_data, dF])


            #DC clusters
            DCc, FOV_data = countDC_clusters(FOV_data, min_cluster_size = 4, max_dist = 30*um_to_pixel)
            DC_clust += DCc
            
            #Neighbor ratio
            center_out_data, peri_out_data = get_proxi_ratio(FOV_data, global_con, bool_dict, center_cells, peri_cells, proxi_dist = 30*um_to_pixel)

            Single_Neigh_ratio_dict = {ct1: {ct2: ((peri_out_data[ct1][ct2][0] / center_out_data[ct1][0]) / (peri_out_data[ct1][ct2][1] / center_out_data[ct1][1]) if center_out_data[ct1][0] != 0 and center_out_data[ct1][1] != 0 and peri_out_data[ct1][ct2][0] != 0 and peri_out_data[ct1][ct2][1] != 0 else np.nan)for ct2 in peri_cells} for ct1 in center_cells}

            CP_dict = {}
            for ctA in center_cells:
                for ctB in center_cells:
                    ct = 'X'.join([ctA, ctB])
                    CP_dict['&'.join(['Neighbor_ratio', ct])] = Single_Neigh_ratio_dict[ctA][ctB]
                    
            single_FOV_dict[FOV] = CP_dict
            
            if c == 0:
                all_center_data = center_out_data
                all_peri_data = peri_out_data

            else:
                all_center_data = {x: [sum(i) for i in zip (all_center_data.get(x, [0, 0]), center_out_data.get(x, [0, 0]))] for x in center_cells} 
                all_peri_data = {x:{y: [sum(i) for i in zip(all_peri_data.get(x, [0, 0]).get(y, [0, 0]), peri_out_data.get(x, [0, 0]).get(y, [0, 0]))] for y in peri_cells} for x in center_cells}


            
            #Tumor area
            T, Tnodes, fig, ax, fig2, ax2, QC_im_name1, QC_im_name2 = get_Tumor_area(FOV_data, FOV, sample,
                                                                          im_path = f'{qsub_im_dir}/{aug_sample}/qsuboptred_{sample}_cycle7_w4_{red_FOV}_t1.TIF',
                                                                          save_dir = save_dir,
                                                                          cell_dist = 30*um_to_pixel,
                                                                          min_cluster_size = 7,
                                                                          degree_of_buffering = 7*um_to_pixel,
                                                                          granularity = 0.01)

            Tumor_polys[FOV] = T

            #Cell location and distances
            FOV_data, areas = MeaAs_boundary_dist(FOV_data, fig, ax, fig2, ax2, QC_im_name1, QC_im_name2,
                                                  Tumor_data = T,
                                                  granularity = 0.006,
                                                  degree_of_buffering = 7*um_to_pixel)

            all_areas += areas
            
            
            #Combine the data into cross FOV dictionaries
            if c == 0:
                out_data = FOV_data

            else:
                out_data = pd.concat([out_data, FOV_data])
                
            c += 1
                    
        except Exception as e:
            print(e)
            plt.close('all')
            fail_FOV.append(FOV)
            continue
        
    #Calculate final metrics
    #Neigh_ratio_dict = {ct1: {ct2: ((all_peri_data[ct1][ct2][0] / all_center_data[ct1][0]) / (all_peri_data[ct1][ct2][1] / all_center_data[ct1][1]) if all_center_data[ct1][0] != 0 and all_center_data[ct1][1] != 0 and all_peri_data[ct1][ct2][0] != 0 and all_peri_data[ct1][ct2][1] != 0 else np.nan)for ct2 in peri_cells} for ct1 in center_cells}

    Neigh_ratio_dict = {}
    for ct1 in center_cells:
        Neigh_ratio_dict[ct1] = {}
        for ct2 in peri_cells:

            #Get number of cells of type 2 (peripheral cells)
            ct_list = ct2.split('_')
            props_list = ['-'.join([prop[:-3], 'status']) for prop in ct_list[1:]]
            sample_data['tempIDneigh'] = sample_data[['identity', *props_list]].apply(lambda x: '_'.join(x.dropna().astype(str)), axis=1)
            ct_filter = sample_data['tempIDneigh'] == ct2
            ctnum = ct_filter.sum()
            
            if ctnum < 10 or all_center_data[ct1][0] == 0:
                Neigh_ratio_dict[ct1][ct2] = 'LowCount'

            elif all_peri_data[ct1][ct2][0] == 0 and ctnum >= 10 and all_center_data[ct1][0] > 0:

                ratio_string1 = (1 / all_center_data[ct1][0]) / (all_peri_data[ct1][ct2][1] / all_center_data[ct1][1])
                Neigh_ratio_dict[ct1][ct2] = 'ZeroC-' + str(ratio_string1)

            elif all_peri_data[ct1][ct2][1] == 0 and ctnum >= 10 and all_center_data[ct1][0] > 0:

                ratio_string1 = (all_peri_data[ct1][ct2][0] / all_center_data[ct1][0]) / (1 / all_center_data[ct1][1])
                Neigh_ratio_dict[ct1][ct2] = 'ZeroP-' + str(ratio_string1)

            elif all_peri_data[ct1][ct2][1] > 0 and all_peri_data[ct1][ct2][0] > 0 and ctnum >= 10 and all_center_data[ct1][0] > 0:

                Neigh_ratio_dict[ct1][ct2] = 'Good-' + str((all_peri_data[ct1][ct2][0] / all_center_data[ct1][0]) / (all_peri_data[ct1][ct2][1] / all_center_data[ct1][1]))


    all_areas = all_areas / (um_to_pixel**2)
    
    proc_FOV = c / len(FOV_list)
                    
    return out_data, all_areas, DC_clust, Neigh_ratio_dict, proc_FOV, fail_FOV, single_FOV_dict, Full_TLS, Partial_TLS



##PROCESSING

m_list = prepare('path_of_choice')

out = gather_all_metrics(sample_list, joined_samples,
                     QC_im_dir='path_of_choice', 
                     qsub_im_dir='path_of_choice',
                     metrics_list='path_of_choice')
