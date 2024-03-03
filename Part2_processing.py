'''
Processing code for part 2 of main HNSCC image analysis pipeline: Alignment, quench subtraction, segmentation and data extraction
authors: Jan Hoelzl, Hannah Peterson

Center for Systems Biology
Massachusetts General Hospital
'''

###IMPORTS

from deepcell.applications import Mesmer
from skimage.filters import threshold_mean
from PIL import Image
from skimage import draw
from scipy import ndimage

import matplotlib.pyplot as plt
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import math
import skimage
import cv2


##VARIABLES TO SET

#parent directory
data_dir = '/home/jupyter/BetterCPS_data/'

#for all samples uncomment the following
sample_list = glob.glob(data_dir+'ill_corrected_patient_data/*') # list of all sample paths

#for a selection of samples uncomment the following
sample_list = [data_dir+'ill_corrected_patient_data/PIO31', ...] #list of direct filepaths to sample folder containing data

#list of all positions in the panel with nuclear stains (measure fluorescence intensity only in nuclear area)
nuclear_stains = ['cycle2_w2', 'cycle2_w3', 'cycle2_w4', 'cycle6_w4'] 

#Scale factor
scale = .1

#Create Mesmer instance
app = Mesmer()

#Dictionary for direct QC, will contain fraction of remaining FOV area after post-alignment cropping
fractions = {}


###PROCESSING

#Iterate through samples
for sam in sample_list:
    
    curr_sam = sam.split('/')[-1]
    if ('_' in curr_sam) and ('PIO' in curr_sam):
        red_curr_sam = curr_sam.split('_')[0]
    else:
        red_curr_sam = curr_sam

    #Status message
    print('Currently processing ', curr_sam)
    
    #Create directories for saved data
    if os.path.isdir(data_dir+'data_processed/data_aligned/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_aligned/'+ curr_sam)
    if os.path.isdir(data_dir+'data_processed/data_crop_qsub/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_crop_qsub/'+ curr_sam)  
    if os.path.isdir(data_dir+'data_processed/data_crop_qsub_opt/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_crop_qsub_opt/'+ curr_sam)  
    if os.path.isdir(data_dir+'data_processed/data_crop_align/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_crop_align/'+ curr_sam)  
    if os.path.isdir(data_dir+'data_processed/data_crop_qsub_red/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_crop_qsub_red/'+ curr_sam)   
    if os.path.isdir(data_dir+'data_processed/data_seg_masks_red/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_seg_masks_red/'+ curr_sam)
    if os.path.isdir(data_dir+'data_processed/data_seg_masks/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/data_seg_masks/'+ curr_sam)
    if os.path.isdir(data_dir+'data_processed/quant_data/'+ curr_sam) == False:
        os.makedirs(data_dir+'data_processed/quant_data/'+ curr_sam)
        
    #Get all cycle paths
    cycle_list = glob.glob(sam+'/*') 
    cycle_list.sort()

    #Get all DAPI image paths from cycle 0 to be used as reference images for alignment
    c0_w1_FOV = glob.glob(cycle_list[0]+'/*w1*.TIF')
    c0_w1_FOV.sort()
    
    #Initialize output
    props_sam = pd.DataFrame()
    FOV_list = []
    Nuc_mask_dict = {}
    Cell_mask_dict = {}
    misaligned = []
    misaligned_pos = []
    MI_QC = None
    MI_ENT_QC = None
    
    sam_fractions = {}

    #Iterate through FOVs
    for ref_file in c0_w1_FOV:
        
        oLmin = False
        meanMin = False
        
        #Usage of try-except construct to avoid exception in case of non-convergence during alignment and handle other exceptions
        
        try: 
            
            #Load reference image
            ref_im = cv2.imread(ref_file,cv2.IMREAD_ANYDEPTH).astype(np.float32)
            ref_file_root = ref_file.split('cycle0_w1_')[-1]
            
            #Handle empty (or almost empty) FOVs with mean fluorescence threshold of 200
            if np.mean(ref_im) < 200:
                
                meanMin = True
                sam_fractions[ref_file_root.split('.')[0]] = ["Mean intensity below threshold", np.mean(ref_im)]
                raise Exception('Exception: Intensity below threshold of 200 in position '+ref_file_root.split('.')[0]+'. Ignoring FOV.')
            
            else:
                
                fov_files = []
                alignment = []

                #Iterate through cycles
                for cy in cycle_list[1:]:
                   
                    #Get all channels
                    chan_files = glob.glob(cy+'/*_'+ref_file_root)  #chan_files = all files from all channels for one FOV
                    chan_files.sort()
                    fov_files.append(chan_files[0]) #record cycle 0 DAPI file as reference of alignment

                    #Alignment
                    temp_im = cv2.imread(chan_files[0],cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    temp_rot, warp_matrix = euc_align(skimage.transform.rescale(ref_im,scale=scale), skimage.transform.rescale(temp_im,scale=scale))
                    
                    #Apply to all channels
                    for chan in chan_files[0:]: 
                        
                        chan_temp = cv2.imread(chan,cv2.IMREAD_ANYDEPTH).astype(np.float32)
                        chan_rot = cv2.warpAffine(chan_temp, warp_matrix*[[1, 1, 1/scale], [1, 1, 1/scale]], (temp_im.shape[1],temp_im.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
                        
                        #Save
                        filename = data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_'+chan.split('/')[-1]
                        Image.fromarray(chan_rot[:,:].astype(np.uint32)).save(filename,format='TIFF')

                    #Record alignment information
                    alignment.append([int(round(warp_matrix[1,-1])),int(round(warp_matrix[0,-1])),np.rad2deg(np.arccos(warp_matrix[0,0])),chan_files[0]]) #yshift, xshift, rotation (in deg), filename

                aligned = pd.DataFrame(alignment,columns=['shifty', 'shiftx', 'deg_rot', 'filename'])

                #Crop to reduce FOVs to the area present in all cycles                               
                #First, find area of tissue that is consistent between all images
                xmax = int(aligned.shiftx.max()/scale)
                xmin = int(aligned.shiftx.min()/scale)
                ymax = int(aligned.shifty.max()/scale)
                ymin = int(aligned.shifty.min()/scale)
                degxy = np.abs(int(np.tan(np.deg2rad(aligned.deg_rot.max()))*ref_im.shape[0]))

                if xmin >= 0:
                    xmin = 0
                if xmax < 0:
                    xmax = 0
                if ymin >= 0:
                    ymin = ref_im.shape[0]
                if ymax < 0:
                    ymax = 0

                #Adjust for rotation and FOV overlap during imaging and calculate edge coordinates
                coordymin = np.maximum((-ymin + degxy), 173)
                coordxmin = np.maximum((-xmin + degxy), 173)
                coordymax = -ymax - degxy
                coordxmax = -xmax - degxy

                #Get overlapping part of reference image
                overlap = ref_im[coordymin:coordymax,coordxmin:coordxmax]
                
                #Raise exception if overlap < 0.2
                if (overlap.size/ref_im.size) < 0.2:
                    
                    oLmin = True
                    sam_fractions[ref_file_root.split('.')[0]] = ["Overlap too low", (overlap.size/ref_im.size)]
                    raise Exception('Exception: Overlap size is below 1/5 in position '+ref_file_root.split('.')[0]+'. Ignoring FOV.')
        
        #Handle other exceptions (non-convergence) during alignment   
        except Exception as ex1:
            
            print(ex1)           
            if oLmin == False and meanMin == False:
                
                print('Exception: Alignment error in position '+ref_file_root.split('.')[0]+'. Ignoring FOV.')
                sam_fractions[ref_file_root.split('.')[0]] = ["Alignment error", -1]
            
        #After correct alignment without raised exceptions, perform cell segmentation using Mesmer             
        else:
            
            #Segmentation array to use as input for Mesmer
            im_co = np.zeros((1,overlap.shape[0],overlap.shape[1],2))

            #Patient samples
            if "PIO" in curr_sam:

                #Use cycle 7 DAPI as nuclear reference to automatically ignore cells lost during imaging
                DAPI_cycle7 = cv2.imread(data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_' + red_curr_sam + '_cycle7_w1_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                DAPI_cycle7 = DAPI_cycle7[coordymin:coordymax,coordxmin:coordxmax]
                DAPI_cycle7 = (DAPI_cycle7 / np.max(DAPI_cycle7)) * (2**16)
                
                #Load DAPI into segmentation array
                im_co[0,:,:,0] = DAPI_cycle7
                
                #Create quench subtracted panCK image
                segment_quench_panCK = cv2.imread(data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_' + red_curr_sam + '_quench6_w4_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                segment_stain_panCK = cv2.imread(data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_' + red_curr_sam + '_cycle7_w4_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32) 
                segment_qsub_panCK = cv2.subtract(segment_stain_panCK,segment_quench_panCK)[coordymin:coordymax,coordxmin:coordxmax]
                
                #Create quench subtracted CD45 image
                segment_quench_CD45 = cv2.imread(data_dir + 'ill_corrected_patient_data/' + curr_sam + '/cycle0/'+ red_curr_sam + '_cycle0_w3_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                segment_stain_CD45 = cv2.imread(data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_' + red_curr_sam + '_cycle1_w3_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                segment_qsub_CD45 = cv2.subtract(segment_stain_CD45,segment_quench_CD45)[coordymin:coordymax,coordxmin:coordxmax]

                #Scale quench subtracted images and create CD45 - panCK maximum intensity projection
                standard_stain1 = (np.maximum(segment_qsub_panCK, 0) / np.max(segment_qsub_panCK)) * (2**16)
                standard_stain2 = (np.maximum(segment_qsub_CD45, 0) / np.max(segment_qsub_CD45)) * (2**16)
                Segment_projection = np.max([standard_stain1, standard_stain2],axis=0)
            
            #Tonsils   
            elif 'Tonsil' in curr_sam:

                #Use cycle 1 DAPI as nuclear reference to automatically ignore cells lost during imaging (tonsils were only imaged in cycle 0 and 1)
                DAPI_cycle1 = cv2.imread(data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_' + red_curr_sam + '_cycle1_w1_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                DAPI_cycle1 = DAPI_cycle1[coordymin:coordymax,coordxmin:coordxmax]
                DAPI_cycle1 = (DAPI_cycle1 / np.max(DAPI_cycle1)) * (2**16)
                
                #Load DAPI into segmentation array
                im_co[0,:,:,0] = DAPI_cycle1
                
                #Create quench subtracted CD45 image
                segment_quench_CD45 = cv2.imread(data_dir + 'ill_corrected_patient_data/' + curr_sam + '/cycle0/'+ red_curr_sam + '_cycle0_w3_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                segment_stain_CD45 = cv2.imread(data_dir + 'data_processed/data_aligned/' + curr_sam + '/aligned_' + red_curr_sam + '_cycle1_w3_' + ref_file_root, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                segment_qsub_CD45 = cv2.subtract(segment_stain_CD45,segment_quench_CD45)[coordymin:coordymax,coordxmin:coordxmax]
                
                #Scale image
                Segment_projection = (np.maximum(segment_qsub_CD45, 0) / np.max(segment_qsub_CD45)) * (2**16)
                
            #Save and enter into segmentation array
            im_co[0,:,:,1] = Segment_projection
            Image.fromarray(Segment_projection.astype(np.uint16)).save(data_dir + 'data_processed/data_seg_masks/'+curr_sam+'/Membrane_segment_image_' + ref_file_root, format='TIFF')
                
            #Perform cell segmentation
            try:
                seg_mask = app.predict(im_co, image_mpp=0.5, compartment='both') 

            #Handle failed segmentations
            except Exception as ex2:
                
                print(ex2)
                print('Mesmer failed segmentation for ', ref_file_root.split('.')[0])
                sam_fractions[ref_file_root.split('.')[0]] = ["Segmentation failed", -2]
                continue

            #After correct segmentation, continue processing
            else:
                
                #Add FOV to list of all successfully processed FOVs and record remaining tissue area after cropping
                FOV_list.append(ref_file_root.split('.')[0])
                sam_fractions[ref_file_root.split('.')[0]] = ["Success", (overlap.size/ref_im.size)]
                
                #Prepare masks
                mask_nucs, mask_cell, relabel_nucs, droplist = data_prep_nuc_cell_match(skimage.segmentation.clear_border(seg_mask[0,:,:,1]),skimage.segmentation.clear_border(seg_mask[0,:,:,0]),graphics=False)

                #Save full resolution masks
                Image.fromarray(mask_nucs.astype(np.uint16)).save(data_dir + 'data_processed/data_seg_masks/'+curr_sam+'/masknucs_' + ref_file_root, format='TIFF')
                Image.fromarray(mask_cell.astype(np.uint16)).save(data_dir + 'data_processed/data_seg_masks/'+curr_sam+'/maskcell_' + ref_file_root, format='TIFF')
                Image.fromarray(relabel_nucs.astype(np.uint16)).save(data_dir + 'data_processed/data_seg_masks/'+curr_sam+'/maskrelabelnucs_' + ref_file_root, format='TIFF')

                #Create and save reduced resolution masks
                resize_nucs = Image.fromarray(mask_nucs.astype(np.uint16))
                resize_cell = Image.fromarray(mask_cell.astype(np.uint16))
                resize_relabel_nucs = Image.fromarray(relabel_nucs.astype(np.uint16))

                resize_nucs = resize_nucs.resize((resize_nucs.width//3, resize_nucs.height//3))
                resize_cell = resize_cell.resize((resize_cell.width//3, resize_cell.height//3))
                resize_relabel_nucs = resize_relabel_nucs.resize((resize_relabel_nucs.width//3, resize_relabel_nucs.height//3))

                resize_nucs.save(data_dir + 'data_processed/data_seg_masks_red/'+curr_sam+'/masknucs_' + ref_file_root, format='TIFF')
                resize_cell.save(data_dir + 'data_processed/data_seg_masks_red/'+curr_sam+'/maskcell_' + ref_file_root, format='TIFF')
                resize_relabel_nucs.save(data_dir + 'data_processed/data_seg_masks_red/'+curr_sam+'/maskrelabelnucs_' + ref_file_root, format='TIFF')

                #Save cell mask and relabeled nuclear mask (to match cell labels) as arrays into dictionary
                Nuc_mask_dict[ref_file_root.split('.')[0]] = relabel_nucs.astype(np.uint16)
                Cell_mask_dict[ref_file_root.split('.')[0]] = mask_cell.astype(np.uint16)

                #Perform quench subtraction and cropping throughout all cycles and channels
                
                props_df = pd.DataFrame()
                MI_FOV = np.zeros((14))
                MI_ENT_FOV = np.zeros((14))

                #Status message
                print('Quench subtracting FOV ', ref_file_root.split('.')[0])

                #Iterate through cycles
                for c in np.arange(1,8):

                    #Gets list of all images for one FOV in each iteration
                    fs_files = glob.glob(data_dir+'data_processed/data_aligned/'+curr_sam+'/*cycle'+str(c)+'*_'+ref_file_root)
                    fs_files.sort()

                    #Load quench files for all channels
                    #Cycle 2-7
                    if c != 1: 
                        fq_files = glob.glob(data_dir+'data_processed/data_aligned/'+curr_sam+'/*quench'+str(c-1)+'*_'+ref_file_root)
                        fq_files.sort()

                    #Cycle 1
                    else:                     
                        fq_files = glob.glob(sam+'/cycle0/*_'+ref_file_root)
                        fq_files.sort()
                        base_DAPI = cv2.imread(fq_files[0],cv2.IMREAD_ANYDEPTH).astype(np.float32)[coordymin:coordymax,coordxmin:coordxmax]

                    #Perform quench subtraction and cropping in all channels

                    for w in np.arange(len(fs_files)):

                        #Do alignment QC with DAPI channel
                        if w == 0:
                            
                            #Load and crop images
                            quench = cv2.imread(fq_files[w],cv2.IMREAD_ANYDEPTH).astype(np.float32)[coordymin:coordymax,coordxmin:coordxmax]
                            stain = cv2.imread(fs_files[w],cv2.IMREAD_ANYDEPTH).astype(np.float32)[coordymin:coordymax,coordxmin:coordxmax]

                            #Calculate mutual information with cycle 0 DAPI image
                            stain_MI = mutual_information_2d(stain.ravel(), base_DAPI.ravel(), normalized=True, nbins=round(np.sqrt((stain.size/5))))
                            quench_MI = mutual_information_2d(quench.ravel(), base_DAPI.ravel(), normalized=True, nbins=round(np.sqrt((stain.size/5))))

                            #Record MI data
                            MI_FOV[(c-1)] = stain_MI
                            MI_FOV[(c+6)] = quench_MI

                            #Save cropped DAPI
                            filename_DAPI_fs = data_dir+'data_processed/data_crop_qsub/'+curr_sam+'/cropped'+fs_files[w].split('/')[-1]
                            Image.fromarray(stain.astype(np.uint16)).save(filename_DAPI_fs, format='TIFF')

                            #Calculate and record MI entropy (not used for QC during final processing)
                            stain_MI_ent = MI_entropy(stain, base_DAPI)
                            quench_MI_ent = MI_entropy(quench, base_DAPI)
                            MI_ENT_FOV[(c-1)] = stain_MI_ent
                            MI_ENT_FOV[(c+6)] = quench_MI_ent

                            #Save position if alignment fails for either stain or quench (threshold = 0.06)
                            if stain_MI < 0.06 or quench_MI < 0.06:
                                if ref_file_root not in misaligned:
                                    
                                    print('Position ' + ref_file_root.split('.')[0] + ' failed alignment! Ignoring FOV. Stain, Quench - MI, cycle, channel(-1): ' + str(stain_MI) +', '+ str(quench_MI) +', '+ str(c) +', '+ str(w))
                                    misaligned.append(ref_file_root)
                                    misaligned_pos.append(ref_file_root.split('.')[0])

                        #Create cropped qsub image and save file for non DAPI channels (including misaligned FOVs)
                        else: 
                            
                            #Load images
                            quench = cv2.imread(fq_files[w],cv2.IMREAD_ANYDEPTH).astype(np.float32)[coordymin:coordymax,coordxmin:coordxmax]
                            stain = cv2.imread(fs_files[w],cv2.IMREAD_ANYDEPTH).astype(np.float32)[coordymin:coordymax,coordxmin:coordxmax]
                            
                            #Create and save quench subtracted image
                            qsub_float = cv2.subtract(stain,quench)
                            qsub = np.maximum(qsub_float, 0)
                            filename_qs = data_dir+'data_processed/data_crop_qsub/'+curr_sam+'/qsub'+fs_files[w].split('/')[-1]
                            Image.fromarray(qsub.astype(np.uint32)).save(filename_qs, format='TIFF')
                            
                            #Save cropped cycle image (non-subtracted)
                            filename_crop_align = data_dir+'data_processed/data_crop_align/'+curr_sam+'/crop_'+fs_files[w].split('/')[-1]
                            Image.fromarray(stain.astype(np.uint32)).save(filename_crop_align, format='TIFF')
                            
                            #Save cropped quench image (non-subtracted)
                            filename_crop_quench = data_dir+'data_processed/data_crop_align/'+curr_sam+'/crop_'+fq_files[w].split('/')[-1]
                            Image.fromarray(quench.astype(np.uint32)).save(filename_crop_quench, format='TIFF')
                            
                            #Save cropped cycle 0 image in reduced resolution for autofluorescence thresholding
                            if c == 1:
                                auto_red = Image.fromarray(quench.astype(np.uint32))
                                auto_red = auto_red.resize((auto_red.width//3, auto_red.height//3))
                                auto_red.save(data_dir+'data_processed/data_crop_qsub_red/'+curr_sam+'/crop_'+fq_files[w].split('/')[-1], format='TIFF')
                           
                #Update QC metrics
                if MI_QC is None:
                    MI_QC = MI_FOV
                    MI_ENT_QC = MI_ENT_FOV

                else:
                    MI_QC = np.append(MI_QC, MI_FOV) 
                    MI_ENT_QC = np.append(MI_ENT_QC, MI_ENT_FOV) 
            
    #After last FOV is processed -->

    #Load QC data into dataframes
    QC_file1 = pd.DataFrame(MI_QC.reshape((len(FOV_list), 14)))
    QC_file1.index = FOV_list
    QC_file2 = pd.DataFrame(MI_ENT_QC.reshape((len(FOV_list), 14)))
    QC_file2.index = FOV_list

    #Gather the 20 FOVs with top mean MI across cycles for image correction by MI minimization
    QC_for_opt = QC_file1.copy()
    QC_for_opt.columns = ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'cycle6', 'cycle7',
                          'cycle0', 'quench1', 'quench2', 'quench3', 'quench4', 'quench5', 'quench6']
    
    QC_for_opt = QC_for_opt.drop(columns=['cycle0'])
    QC_for_opt['meanMI'] = QC_for_opt.mean(axis=1)
    QC_for_opt = QC_for_opt.sort_values(by=['meanMI'], ascending=False)
    TopMi_FOV = list(QC_for_opt.index)[0:20]

    #Create corrected quench-subtracted images

    cycleNames = ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'cycle6', 'cycle7']
    all_alphas = {}
    all_final_alphas = {}

    #Perform corrections for each channel individually
    for chan in range(2,5):
        alpha_dict = {}
        
        #Iterate through the 20 best aligned FOVs
        for FOV in TopMi_FOV:
            
            #Load all images for a given FOV
            FOV_imDict = {}
            for cycle in cycleNames:
                
                img_name = data_dir+'data_processed/data_crop_qsub/'+curr_sam+'/qsubaligned_'+red_curr_sam+'_'+cycle+'_w'+str(chan)+'_'+FOV+'.TIF'
                FOV_imDict[cycle] = cv2.imread(img_name,cv2.IMREAD_ANYDEPTH).astype(np.int32)
            
            #Collect optimized alpha parameters for a given FOV
            alpha_dict[FOV] = Minimize_MI(FOV_imDict)

        #Get final alpha values across the 20 analyzed FOVs
        final_alpha = median_alpha(alpha_dict)
        all_alphas['_'.join(['w', str(chan)])] = alpha_dict
        all_final_alphas['_'.join(['w', str(chan)])] = final_alpha

        #Apply corrections to all images
        for FOV in FOV_list:
            
            #Load all images for a given FOV
            FOV_imDict = {}
            for cycle in cycleNames:
                img_name = data_dir+'data_processed/data_crop_qsub/'+curr_sam+'/qsubaligned_'+red_curr_sam+'_'+cycle+'_w'+str(chan)+'_'+FOV+'.TIF'
                FOV_imDict[cycle] = cv2.imread(img_name,cv2.IMREAD_ANYDEPTH).astype(np.int32)
            
            #Get dictionary of corrected images
            corrected = Correct_im(FOV_imDict, final_alpha)
            
            for c, im in corrected.items():
                
                #Save full resolution corrected image
                out_name = data_dir+'data_processed/data_crop_qsub_opt/'+curr_sam+'/qsubopt_'+red_curr_sam+'_'+c+'_w'+str(chan)+'_'+FOV+'.TIF'
                Image.fromarray(im.astype(np.uint32)).save(out_name, format='TIFF')
                
                #Save image with reduced resolution
                qsub_red = Image.fromarray(im.astype(np.uint32))
                qsub_red = qsub_red.resize((qsub_red.width//3, qsub_red.height//3))
                qsub_red.save(data_dir+'data_processed/data_crop_qsub_red/'+curr_sam+'/qsuboptred_'+red_curr_sam+'_'+c+'_w'+str(chan)+'_'+FOV+'.TIF', format='TIFF')
                              
    
    #Perform measurements

    #Iterate through cycles and channels            
    for c in np.arange(1,8):
        for chan in np.arange(2,5):

            #Create basename string
            basename = 'cycle'+str(c)+'_w'+str(chan)

            #Get all qsub files
            qsub_files_all = glob.glob(data_dir+'data_processed/data_crop_qsub_opt/'+curr_sam+'/qsub*'+basename+'*.TIF')
            
            #Filter out misaligned FOVs
            qsub_files = list(filter(lambda x: all(map(lambda y: y not in x, misaligned)), qsub_files_all))
            qsub_files.sort()

            #Measurements for each FOV individually
            for image in qsub_files:
                
                #Load qsub image
                qsub_img = cv2.imread(image,cv2.IMREAD_ANYDEPTH).astype(np.uint32)
                position = image.split('_')[-2] + '_t1'

                #Load quench image
                if c == 1:
                    quench_img_path = data_dir+'data_processed/data_crop_align/'+curr_sam+'/crop_'+red_curr_sam+'_cycle0_'+'_'.join(image.split('_')[-3:])
                else:
                    quench_img_path = data_dir+'data_processed/data_crop_align/'+curr_sam+'/crop_aligned_'+red_curr_sam+'_quench'+str(c-1)+'_'+'_'.join(image.split('_')[-3:])
                quench_img = cv2.imread(quench_img_path,cv2.IMREAD_ANYDEPTH).astype(np.uint32)
                
                #Load cycle image
                cycle_img_path = data_dir+'data_processed/data_crop_align/'+curr_sam+'/crop_aligned_'+red_curr_sam+'_cycle'+str(c)+'_'+'_'.join(image.split('_')[-3:])
                cycle_img = cv2.imread(cycle_img_path,cv2.IMREAD_ANYDEPTH).astype(np.uint32)

                #Regionprops measurements using either cell or nuclear mask (depending on marker)               
                if basename in nuclear_stains:
                    hp_props_dict = skimage.measure.regionprops_table(Nuc_mask_dict[position],qsub_img,separator='_',properties=['label','mean_intensity','area','centroid', 'equivalent_diameter'], extra_properties=(p50,))
                else:
                    hp_props_dict = skimage.measure.regionprops_table(Cell_mask_dict[position],qsub_img,separator='_',properties=['label','mean_intensity','area','centroid', 'equivalent_diameter'], extra_properties=(p50,))
                try:
                    if basename in nuclear_stains:
                        extra_measure = skimage.measure.regionprops_table(Nuc_mask_dict[position],qsub_img, separator='_',properties=['image', 'intensity_image', 'weighted_centroid'])
                    else:
                        extra_measure = skimage.measure.regionprops_table(Cell_mask_dict[position],qsub_img, separator='_',properties=['image', 'intensity_image', 'weighted_centroid'])
                except:
                    print('Extra properties cannot be measured. Check Scikit-image version!')

                #Prepare column names
                col_name_mean = '_'.join(['mean_intensity', basename])
                col_name_centrality_ratio = '_'.join(['centrality_ratio', basename])
                col_name_centrality_tendency = '_'.join(['centrality_tendency', basename])
                col_name_centrality_spear = '_'.join(['centrality_spear', basename])
                col_name_IQ0109mean = '_'.join(['IQmean_0.1-0.9', basename])
                col_name_IQ005095mean = '_'.join(['IQmean_0.05-0.95', basename])
                col_name_skewness = '_'.join(['skewness', basename])
                col_name_median = '_'.join(['median_intensity', basename])
                col_name_prop_50 = '_'.join(['prop_50', basename])
                col_name_MI_stain = '_'.join(['MI_stain', basename])
                col_name_MI_ENT_stain = '_'.join(['MI_ENT_stain', basename])
                col_name_MI_quench = '_'.join(['MI_quench', basename])
                col_name_MI_ENT_quench = '_'.join(['MI_ENT_quench', basename])
                col_name_artifact = '_'.join(['artifact', basename])

                #Load first measurements into dataframe
                props = pd.DataFrame(hp_props_dict)
                props.columns = ['label', col_name_mean, 'area', 'centroid_x', 'centroid_y', 'equivalent_diameter', col_name_median]

                #Calculate inter-quantile means
                props[col_name_IQ0109mean] = IQMean(Cell_mask_dict[position], qsub_img, qL=0.1, qU=0.9)
                props[col_name_IQ005095mean] = IQMean(Cell_mask_dict[position], qsub_img, qL=0.05, qU=0.95)

                #Calculate centrality of stain metrics (not used in downstream analysis)
                ratio, tendency, spear = centrality_of_stain(extra_measure['image'], extra_measure['intensity_image'])
                props[col_name_centrality_ratio] = ratio
                props[col_name_centrality_tendency] = tendency
                props[col_name_centrality_spear] = spear

                #Calculate skewness of stain
                props[col_name_skewness] = skewness_of_stain(extra_measure['weighted_centroid_0'], extra_measure['weighted_centroid_1'],
                                                                             hp_props_dict['centroid_0'], hp_props_dict['centroid_1'], 
                                                                             hp_props_dict['equivalent_diameter'])

                #Calculate prop_50 (entropy) metric
                props[col_name_prop_50] = prop_ness(extra_measure['image'], extra_measure['intensity_image'], prop = 0.5)

                props['FOV'] = position

                #QC data
                props['overlap'] = sam_fractions[position][1]
                props[col_name_MI_stain] = QC_file1.at[position, (c-1)]
                props[col_name_MI_ENT_stain] = QC_file2.at[position, (c-1)]
                props[col_name_MI_quench] = QC_file1.at[position, (c+6)]
                props[col_name_MI_ENT_quench] = QC_file2.at[position, (c+6)]
                
                #Annotate artifacts
                props[col_name_artifact] = annotate_artifacts(Cell_mask_dict[position], quench_img, cycle_img, int_thres_global=100, int_thres_local=700, ratio_thres=6000, rad=3)

                #Measure autofluorescence in cycle 0
                if c == 1:
                    col_name_autofluo = '_'.join(['autofluo', basename])
                    autofluorescence_measure = skimage.measure.regionprops_table(Cell_mask_dict[position],quench_img,separator='_',properties=['label','mean_intensity'])
                    props[col_name_autofluo] = autofluorescence_measure['mean_intensity']

                #Update output data
                if image == qsub_files[0]:
                    props_df = props
                else:
                    props_df = props_df.append(props)

            #Create unique IDs for all cells
            props_df['unique_ident'] = [str(x) + '_' + str(y) for x, y in zip(props_df['FOV'], props_df['label'])]

            #Update output data
            if (c == 1) & (chan == 2):               
                props_sam = props_df
            
            else:
                #Ensure misaligned FOVs are dropped
                props_sam = props_sam.loc[~props_sam['FOV'].isin(misaligned_pos), : ]
                props_df = props_df.loc[~props_df['FOV'].isin(misaligned_pos), : ]

                #Merge dataframe along 'unique_ident'
                if len(props_sam) != 0 and c != 1:
                    props_sam = pd.merge(props_sam, props_df[['unique_ident', col_name_mean, col_name_IQ0109mean, col_name_IQ005095mean, col_name_centrality_ratio, col_name_centrality_tendency, col_name_centrality_spear, col_name_skewness, col_name_median, col_name_prop_50, col_name_MI_stain, col_name_MI_ENT_stain, col_name_MI_quench, col_name_MI_ENT_quench, col_name_artifact]],on='unique_ident')
                elif len(props_sam) != 0 and c == 1:
                    props_sam = pd.merge(props_sam, props_df[['unique_ident', col_name_mean, col_name_IQ0109mean, col_name_IQ005095mean, col_name_centrality_ratio, col_name_centrality_tendency, col_name_centrality_spear, col_name_skewness, col_name_median, col_name_prop_50, col_name_MI_stain, col_name_MI_ENT_stain, col_name_MI_quench, col_name_MI_ENT_quench, col_name_artifact, col_name_autofluo]],on='unique_ident')
                else:
                    print('Warning! No data measured. (Check alignments)')               
                
    #Save QC data (MI and MI entropy)
    QC_file1.columns = ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'cycle6', 'cycle7',
                        'cycle0', 'quench1', 'quench2', 'quench3', 'quench4', 'quench5', 'quench6']
    QC_file2.columns = ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'cycle6', 'cycle7',
                        'cycle0', 'quench1', 'quench2', 'quench3', 'quench4', 'quench5', 'quench6']
    
    QC_file1.to_csv(data_dir+'data_processed/quant_data/'+ curr_sam+'/MI_QC.csv')
    QC_file2.to_csv(data_dir+'data_processed/quant_data/'+ curr_sam+'/MI_ENT_QC.csv')
    
    #Save complete measured data
    props_sam.to_csv(data_dir+'data_processed/quant_data/'+curr_sam+'/Data.csv', index=False)
    
    ##Save fractions to dictionary for direct access
    fractions[curr_sam] = sam_fractions

    #Save logging files
    Logging_file_name = data_dir+'data_processed/quant_data/'+curr_sam+'/logging_fractions.txt'
    All_alpha_file_name = data_dir+'data_processed/quant_data/'+curr_sam+'/logging_all_alpha.txt'
    Final_alpha_file_name = data_dir+'data_processed/quant_data/'+curr_sam+'/logging_final_alpha.txt'
    
    with open(Logging_file_name, 'wb') as handle:
        pickle.dump(sam_fractions, handle)
    with open(All_alpha_file_name, 'wb') as handle:
        pickle.dump(all_alphas, handle)
    with open(Final_alpha_file_name, 'wb') as handle:
        pickle.dump(all_final_alphas, handle)
        