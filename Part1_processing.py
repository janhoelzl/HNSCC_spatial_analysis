'''
Processing code for part 1 of main HNSCC image analysis pipeline: Illumination correction
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''

##IMPORTS

from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

import numpy as np
import pandas as pd
import networkx as nx
import pybasic
import skimage
import glob
import os
import cv2
import shutil
import pickle



##VARIABLES TO SET

#Dictionary of imaging batches to be processed as keys, with the lists of samples in each of them as values
batch_list = {'16': ['PIO31', 'PIO21', 'PIO39_1', 'PIO39_2', 'Tonsil_07122023']}

#Parent directory
data_dir = ''

#List of all sample paths
sample_list = glob.glob(data_dir+'data_raw/*')

#List of imaging channel abbreviations
channel_list = ['_w2_', '_w3_', '_w4_']

#QC variables (see extended methods)
MI_thres = 0.10
img_num_lim = 60



##PROCESSING

#Iterate through batches
for nr, batch in batch_list.items():

    #Create directory to temporarily store correction images for each batch
    tempdir = data_dir+'temp'
    if os.path.isdir(tempdir) == False:
        os.makedirs(tempdir)
        
    #Get list of cycles
    cycle_list = glob.glob(sample_list[1]+'/*')
    cycle_list.sort()
    
    #Iterate through cycles
    for cycle in cycle_list:

        cycle = cycle.split('/')[-1]
        
        #Create dictionary to store final mean correction matrices
        corr_outdir = data_dir+'processing_log_data/PyBaSiC_new/Mean_corr_matrices_batch' + str(nr) + '/' + cycle + '/'
        if os.path.isdir(corr_outdir) == False:
            os.makedirs(corr_outdir)

        #Iterate through channels
        for chan in channel_list:
            
            #Create dicts to save correction matrices
            corr_dict_flat = {}
            corr_dict_dark = {}
            data_dict = {}
            len_dict = {}

            #Copy files to tempdir sample-wise
            for s in batch:
                
                #Delete old sample's images
                exist_temp = glob.glob(tempdir + '/*')

                if len(exist_temp) != 0:
                    for file in exist_temp:
                        os.remove(file)
                
                #Skip tonsils for creation of correction matrices
                if s.startswith('Tonsil'):
                    continue

                else:
                    all_files = glob.glob(data_dir + 'data_raw/' + s + '/' + cycle + '/*.TIF')
                    if (len(all_files) * (1/4)) > img_num_lim:
                        for file in all_files:
                            #Do not use DAPI for correction
                            if '_w1_' not in file and chan in file:
                                new_filename = tempdir + '/sample_' + file.split('/')[-1]
                                shutil.copy(file, new_filename)

                    #Skip sample if number of FOVs below QC threshold
                    else:
                        continue

                #Calculate sample-wise correction matrices using PyBaSiC
                images = pybasic.tools.load_data(tempdir, '.TIF', verbosity = False)
                flatfield, darkfield = pybasic.basic(images, darkfield=True, max_reweight_iterations=15, max_iterations=800)

                len_dict[s] = len(images)

                #Create dictionary to store sample-wise correction matrices
                check_outdir = data_dir+'processing_log_data/PyBaSiC_new/' + s + '/' + cycle + '/'
                if os.path.isdir(check_outdir) == False:
                    os.makedirs(check_outdir)
                    
                #Save sample-wise correction matrices to disk and dict
                Image.fromarray(flatfield.astype(np.float32)).save(f'{check_outdir}flatfield{chan}.TIF', format='TIFF')   
                Image.fromarray(darkfield.astype(np.float32)).save(f'{check_outdir}darkfield{chan}.TIF', format='TIFF')
                
                if not np.isnan(flatfield).all():
                    corr_dict_flat[s] = flatfield
                if not np.isnan(darkfield).all():
                    corr_dict_dark[s] = darkfield
                
            #Calculate mutual information for QC purposes
            MI_dict_flat = {}
            for sam1 in corr_dict_flat.keys():
                MI_subdict_flat = {}
                for sam2 in corr_dict_flat.keys():
                    try:
                        MI_subdict_flat[sam2] = mutual_information_2d(corr_dict_flat[sam1].ravel(), corr_dict_flat[sam2].ravel())
                    except:
                        MI_subdict_flat[sam2] = 0
                        
                MI_dict_flat[sam1] = MI_subdict_flat

            MI_df_flat = pd.DataFrame(MI_dict_flat)

            poss_sams = np.array(MI_df_flat.index.to_list())
            
            data_dict['batch_samples'] = batch
            data_dict['MI_flat'] = MI_df_flat
                    
            #Find biggest connected cluster(s) > MI_thres
            graph_arr = np.array(MI_df_flat)
            graph_arr[graph_arr<=MI_thres] = 0
            graph_arr[graph_arr>MI_thres] = 1
            graph = nx.from_numpy_matrix(graph_arr)
            clusters = list(nx.enumerate_all_cliques(graph))
            connected_clusters = [li for li in clusters if len(li) == len(max(clusters, key=len))]
            
            #Determine samples to use for final connection matrix
            if len(connected_clusters) > 1:

                MI_arrs = {i: MI_df_flat.iloc[connected_clusters[i], connected_clusters[i]].values for i in range(0,len(connected_clusters),1)}

                for ind, array in MI_arrs.items():
                    temp_array = array
                    temp_array[np.identity(temp_array.shape[0]) == 1] = 0
                    MI_arrs[ind] = temp_array

                cluster_MIs = {key: np.mean(arr) for key, arr in MI_arrs.items()}
                winning_clusters = [key for key, value in cluster_MIs.items() if value == max(cluster_MIs.values())]
                
                data_dict['conflicts'] = 'Multiple equal lenght clusters present'
                data_dict['MI_conflict_data'] = cluster_MIs

                if len(winning_clusters) == 1:
                    good_sams = poss_sams[connected_clusters[winning_clusters[0]]]

                else:
                    good_sams = np.array([max(len_dict, key=len_dict.get)])
                    
            elif len(connected_clusters) == 1:
                good_sams = poss_sams[connected_clusters[0]]
                data_dict['conflicts'] = 'Single winning cluster present'
                data_dict['MI_conflict_data'] = 'X'
                
            data_dict['used_samples'] = good_sams
                    
            #Calculate mean correction matrices
            mean_flat = corr_dict_flat[good_sams[0]]
            mean_dark = corr_dict_dark[good_sams[0]]
            for sample in good_sams[1:]:
                mean_flat = mean_flat + corr_dict_flat[sample]
                mean_dark = mean_dark + corr_dict_dark[sample]
                
            mean_flat = mean_flat / len(good_sams)
            mean_dark = mean_dark / len(good_sams)
            
            #Save mean correction matrices
            Image.fromarray(mean_flat.astype(np.float32)).save(f'{corr_outdir}flatfield{chan}.TIF', format='TIFF')   
            Image.fromarray(mean_dark.astype(np.float32)).save(f'{corr_outdir}darkfield{chan}.TIF', format='TIFF')
            
            #Save logging dictionary
            out_dict_name = corr_outdir+'logging_dictionary'+chan+'.txt'
            with open(out_dict_name, 'wb') as file:
                pickle.dump(data_dict, file)
                
            
            #Apply corrections to raw images
            for s in batch:

                if (s.startswith('Tonsil') and cycle not in 'cycle0cycle1'):
                    continue
                  
                #Correct images  
                else:

                    #Create directory for corrected images
                    base_outdir = data_dir+'ill_corrected_patient_data/' + s + '/' + cycle + '/'
                    if os.path.isdir(base_outdir) == False:
                        os.makedirs(base_outdir)

                    all_pos = glob.glob(data_dir + 'data_raw/' + s + '/' + cycle + '/*.TIF')

                    #Iterate through images/FOVs
                    for img in all_pos:

                        if '_w1_' not in img and chan in img:

                            #Correct
                            img_list = [cv2.imread(img,cv2.IMREAD_ANYDEPTH).astype(np.float32)]
                            img_corrected = pybasic.correct_illumination(images_list = img_list, 
                                                                         flatfield = mean_flat, 
                                                                         darkfield = mean_dark)

                            #Ensure non-negativity
                            img_out = np.maximum(img_corrected[0], 0)

                            #Save
                            out_path = base_outdir + img.split('/')[-1]
                            Image.fromarray(img_out.astype(np.uint32)).save(out_path, format='TIFF')

                        #Copy DAPI images non-corrected
                        elif '_w1_' in img:
                            shutil.copy(img, base_outdir)
                        else:
                            pass