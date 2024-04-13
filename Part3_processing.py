'''
Processing code for part 3 of main HNSCC image analysis pipeline: Cell phenotyping
Not included in reproducible run due to interactive elements. Requires user inputs.
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''

##IMPORTS

import os
import sys
import glob
import math
import pickle
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import scipy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.signal import peak_widths

import skimage
from skimage.color import label2rgb
from skimage.io import imread
from skimage import draw

from PIL import ImageTk, ImageEnhance
from PIL import Image as Img

import cv2

from tkinter import *
from tkinter import ttk



##VARIABLES TO SET

#Parent directory
data_dir = 'path_of_choice'

#List of all (sub)sample paths before combining
pre_sample_list = [data_dir+'path_of_choice/PIO1']

#List of all sample paths after combining
sample_list = [data_dir+'path_of_choice/PIO1']

#Dictionary mapping all markers to their their position in the panel (cycle and channel)
marker_dict = {'PD-L1': 'cycle1_w2', 'CD45': 'cycle1_w3', 'PD1': 'cycle1_w4', 
			   'FoxP3': 'cycle2_w2', 'TCF1': 'cycle2_w3', 'p63': 'cycle2_w4', 
			   'CD3': 'cycle3_w2', 'CD163': 'cycle3_w3', 'CD20': 'cycle3_w4',
			   'LAMP3': 'cycle4_w2', 'CD11b': 'cycle4_w3', 'CTLA4': 'cycle4_w4',
			   'CD11c': 'cycle5_w2', 'CD8': 'cycle5_w3', 'CD66b': 'cycle5_w4',
			   'CD14': 'cycle6_w2', 'CD68': 'cycle6_w3', 'KI67': 'cycle6_w4',
			   'Quad': 'cycle7_w2', 'CD4': 'cycle7_w3', 'panCK': 'cycle7_w4'}

#Dictionary mapping all markers to their priority group for thresholding (cytoplasmic have priority over membrane stains)
pheno_markers = {'CD68': 1, 'LAMP3': 1, 'CD4': 2, 'CD11c': 2, 'CD8': 2, 'CD66b': 2, 'CD20': 2, 'CD14': 2, 'CD3': 2, 'CD11b': 2}

#Markers that negative exponential / combined thresholding strategy should be applied to
markers_exp = ['CD4', 'CD11c', 'CD8', 'CD66b', 'CD45', 'PD1', 'CD20', 'CD14', 'CD68', 'CD163', 'CD11b', 'CD3', 'CTLA4', 'LAMP3']

#Pre-set threshold multiplication factors for skewness thresholding
skewness_factors = {'CD4': 1.4, 'CD11c': 1.3, 'CD8': 1.4, 'CD66b': 1.6, 'CD45': 1.3, 'PD1': 1.1, 'CD20': 1.1, 'CD14': 1.3, 'CD68': 1.3, 'CD11b': 1.3, 'CD163': 1.3, 'CD3': 1.4, 'CTLA4': 1.2, 'LAMP3': 1.3}

#Pre-set threshold multiplication factors for intensity thresholding
aberrant_scale = {'CD66b': 1.5}

#Markers that FWHM thresholding should be applied to
markers_FWHM = ['panCK', 'Quad', 'p63', 'KI67', 'FoxP3', 'TCF1']  #Ki67 is treated seperately 

#Markers to use for tumor vs. non-tumor split
markersA = ['CD45', 'panCK', 'Quad', 'p63', 'KI67']

#Other immune markers to be thresholded in two groups (by authors J.O. and J.H.)
markersJan = ['CD20', 'CD8', 'PD1', 'CD4', 'CD3', 'CTLA4']
markersJulie = ['LAMP3', 'CD11b', 'CD11c', 'CD66b', 'CD14', 'CD68']

#Dictionary mapping cell phenotypes to lists containing all possible sets of marker combinations that allow the assignment of the given phenotype
pheno_dict = {#CCR7+ DCs
			  'DC3': [se for se in power_set_alt(items=['LAMP3', 'CD45', 'CD11b', 'CD11c', 'PD1', 'CTLA4'], conds={'CD45', 'CD11b', 'CD11c'}) if 'LAMP3' in se], 
			  
			  #Lymphoid
			  'CD4-Tcell': power_set(items=['CD4', 'CD45', 'CD3', 'PD1', 'CTLA4'], conds={'CD4'}),
			  'CD8-Tcell': power_set(items=['CD8', 'CD45', 'CD3', 'PD1', 'CTLA4'], conds={'CD8'}),
			  'Bcell': power_set(items=['CD20', 'CD45', 'PD1', 'CTLA4'], conds={'CD20'}),

			  #Myeloid
			  'Neutrophil': power_set(items=['CD66b', 'CD11c', 'CD11b','CD45', 'PD1', 'CTLA4'], conds={'CD66b'}),
			  'Macrophage': power_set(items=['CD68', 'CD11c', 'CD11b', 'CD45', 'PD1', 'CTLA4'], conds={'CD68'}),
			  'Monocyte': power_set(items=['CD14', 'CD11c', 'CD11b', 'CD45', 'PD1', 'CTLA4'], conds={'CD14'}),
			  'Other-myeloid': power_set_alt(items=['CD11c', 'CD11b', 'CD45', 'PD1', 'CTLA4'], conds={'CD11b', 'CD11c'}),
	
			  'Other-CD45': power_set(items=['CD45', 'PD1', 'CTLA4'], conds={'CD45'})}

#Static cell size threshold
size_threshold = 400

#All different intensity measurements that are present in the input data and should be shown in QC plots
intens_measurements = ['mean_intensity', 'IQmean_0.05-0.95', 'IQmean_0.1-0.9']

#Intensity measure to use for phenotyping
intensity_measure = 'IQmean_0.05-0.95'



##PROCESSING FUNCTIONS

def prep_plotting():
	
	"""
	Generates necessary directories and produces and saves general diagnostics plots.
	This function handles the preprocessing required by combining samples, creating directories,
	and generating plots. It is meant to be run as a preliminary step before further phenotyping.
	
	It does not take parameters directly; instead, it uses the predefined variables at the start of the script,
	such as sample lists and directory paths. It combines data from multiple images tissue areas from one sample,
	applies cell size thresholding to filter data, and creates a variety of diagnostic plots, including histograms
	and scatter plots for the different intensity measurements and markers.

	Returns
	-------
	None : This function does not return a value. It performs file operations to save all plots directly to the filesystem.
	"""
	
	#Perform subsample combinations
	sam_to_combine = [sam for sam in sample_list if 'comb' in sam]
			
	for sam in sam_to_combine:

		sam = sam.split('/')[-1]
		
		#Make directories
		new_quant = data_dir+'data_processed/quant_data/'+sam
		new_im = data_dir+'data_processed/data_crop_qsub_red/'+sam
		new_seg = data_dir+'data_processed/data_seg_masks_red/'+sam
		os.mkdir(new_quant)
		os.mkdir(new_im)
		os.mkdir(new_seg)
		
		sam_name = sam.split('_')[0]
		sub_sam = [s for s in pre_sample_list if sam_name in s]
		
		#Iterate through subsamples
		for sub in sub_sam:
			
			sub = sub.split('/')[-1]
			nr = sub.split('_')[-1]
			fileend = 't'+nr+'.TIF'
			
			old_im = data_dir+'data_processed/data_crop_qsub_red/'+sub
			old_seg = data_dir+'data_processed/data_seg_masks_red/'+sub
			
			#Copy reduced resolution qsub images to new directory
			for file in os.listdir(old_im):
				splitname = file.split('/')[-1]
				splitname = splitname.split('_')
				splitname[-1] = fileend
				new_name = new_im +'/'+ '_'.join(splitname)
				shutil.copy(old_im+'/'+file, new_name)
			
			#Copy reduced resolution segmentation masks to new directory
			for file in os.listdir(old_seg):
				splitname = file.split('/')[-1]
				splitname = splitname.split('_')
				splitname[-1] = fileend
				new_name = new_seg +'/'+ '_'.join(splitname)
				shutil.copy(old_seg+'/'+file, new_name)
			
			#Create combined measurement data file
			quanti = pd.read_csv(data_dir + 'data_processed/quant_data/' + sub + '/Data.csv')
			
			if nr == '1':
				quant_comb = quanti

			else:
				#Make cell identifiers unique
				replacement = '_t'+nr
				quanti['FOV'] = quanti['FOV'].str.replace('_t1', replacement)
				quanti['unique_ident'] = quanti['unique_ident'].str.replace('_t1', replacement)
				quant_comb = quant_comb.append(quanti)
		
		#Save combined measurement data file
		quant_comb.to_csv(new_quant + '/Data.csv')
		
	all_sample_list = np.unique(sample_list+pre_sample_list)
				
	#Make graphics

	for sam in all_sample_list:

		#Status message
		print('Currently making graphics for ', sam.split('/')[-1])
		
		#Create directories
		if os.path.isdir(data_dir+'data_processed/derivative_data/'+ sam.split('/')[-1]) == False:
			os.makedirs(data_dir+'data_processed/derivative_data/'+ sam.split('/')[-1])
		if os.path.isdir(data_dir+'data_processed/diagnostics/'+ sam.split('/')[-1]) == False:
			os.makedirs(data_dir+'data_processed/diagnostics/'+ sam.split('/')[-1])
		if os.path.isdir(data_dir+'data_processed/phenotype_masks/'+ sam.split('/')[-1]) == False:
			os.makedirs(data_dir+'data_processed/phenotype_masks/'+ sam.split('/')[-1])
		if os.path.isdir(data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]) == False:
			os.makedirs(data_dir+'data_processed/temp_data/'+ sam.split('/')[-1])
	
		#Read data
		input_path = data_dir + 'data_processed/quant_data/' + sam.split('/')[-1] + '/Data.csv'
		input_data = pd.read_csv(input_path)

		#Size thresholding (threshold = 400)
		filter_data = input_data.loc[input_data['area'] > size_threshold, : ]

		#Create and save diagnostic plots

		#Iterate through the different intensity measurements
		for intens_measure in intens_measurements:
			
			#Log2 transformed histogram
			fig = plt.figure(figsize=(30,16))
			gs = fig.add_gridspec(7, 3, hspace=0.5, wspace=0.5)
			ax = gs.subplots(sharex=False, sharey=False)
			i = 0
			for key, value in marker_dict.items():
				ax[i%7][i%3].hist(np.log2(filter_data['_'.join([intens_measure, value])] + 1),
								 bins = 50)
				ax[i%7][i%3].set_title(key)
				ax[i%7][i%3].set_facecolor('xkcd:white')
				i += 1
			plt.savefig(data_dir+'data_processed/diagnostics/'+ sam.split('/')[-1] + '/Histograms_log2.png')
			plt.close('all')

			#Raw histogram
			fig = plt.figure(figsize=(30,16))
			gs = fig.add_gridspec(7, 3, hspace=0.5, wspace=0.5)
			ax = gs.subplots(sharex=False, sharey=False)
			i = 0
			for key, value in marker_dict.items():
				ax[i%7][i%3].hist(filter_data['_'.join([intens_measure, value])],
								  bins = 50)
				ax[i%7][i%3].set_title(key)
				ax[i%7][i%3].set_facecolor('xkcd:white')
				i += 1
			plt.savefig(data_dir+'data_processed/diagnostics/'+ sam.split('/')[-1] + '/Histograms_raw.png')
			plt.close('all')
			
			#Combined scatterplot (showing intensity, skewness and prop_50)
			fig = plt.figure(figsize=(30,16))
			gs = fig.add_gridspec(7, 3, hspace=0.5, wspace=0.5)
			ax = gs.subplots(sharex=False, sharey=False)
			i = 0
			for key, value in marker_dict.items():
				ax[i%7][i%3].scatter(filter_data['_'.join([intens_measure, value])], filter_data['_'.join(['prop_50', value])], 
									 alpha=0.05, c=filter_data['_'.join(['skewness', value])])
				ax[i%7][i%3].set_title(key) 
				ax[i%7][i%3].set_facecolor('xkcd:white')
				i += 1
			plt.savefig(data_dir+'data_processed/diagnostics/'+ sam.split('/')[-1] + '/Scatterplots_' + intens_measure + '_prop50.png')
			plt.close('all')


def pheno_part1(name_params='Tumor_split_model_params.txt'):

	"""
	Performs thresholding to seperate tumor from non-tumor cells in the sample data. 
	It applies quality control (QC), autofluorescence filtering, and intensity thresholding 
	for cell classification based on predefined markers (phenotype_dict). 
	The function iterates through each sample, 
	filters data based on cell area and artifact presence, determines autofluorescence thresholds, 
	and classifies cells using semi-automated, GUI-assisted thresholding.
	The results, including thresholds and model parameters, are saved in intermediary CSV files 
	and a text file containing threshold/model data for downstream processing.

	Parameters
	----------
	name_params : str, optional
		The filename for saving the model parameters. Default is 'Tumor_split_model_params.txt'.

	Returns
	-------
	None : This function does not return a value. It performs file operations to save the intermediary data (post QC) 
	and model parameters (for downstream ID assignment) directly to the filesystem.
	"""
	
	#Iterate through samples
	for sam in sample_list:

		#Status message
		print('Currently performing tumor-non-tumor split for ', sam.split('/')[-1])
		
		patient = sam.split('/')[-1]
		patient = patient.split('_')[0]
		
		#Read data
		input_path = data_dir + 'data_processed/quant_data/' + sam.split('/')[-1] + '/Data.csv'
		input_data = pd.read_csv(input_path)

		#Quality control
		
		#Cell size (threshold = 400)
		filter_data = input_data.loc[input_data['area'] > size_threshold, : ]
		
		#Bright artifacts
		filter_data['artifact_cols'] = filter_data[[col for col in filter_data.columns if 'artifact' in col]].agg('_'.join, axis=1)
		filter_data = filter_data.loc[~filter_data['artifact_cols'].str.contains('Fail', regex=False), ]
		
		#Autofluorescence

		#Create automatic threshold suggestions
		thres_auto = autofluo_thresholding(data = filter_data, 
										   columns_autofluo = ['autofluo_cycle1_w2', 'autofluo_cycle1_w3', 'autofluo_cycle1_w4'], 
										   base_scale_factor=1.4, 
										   base_symmetry='right')

		#Perform manual check with GUI for each channel
		for chan_nr in range(2,5):
			
			#Prepare data to create GUI object
			chan = 'cycle0_w'+str(chan_nr)
			marker = 'autofluo_cycle1_w'+str(chan_nr)
			in_data = filter_data[['FOV', 'label', 'unique_ident', marker]]
			in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity']
			
			FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
			FOV_ranks = FOV_ranks.index.to_list()
			
			basename_auto = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/crop_'+patient+'_'+chan+'_'
			basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
			FOV_dict = {FOV: basename_auto+FOV+'.TIF' for FOV in FOV_ranks}
			seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
			#GUI
			window = GUI_FWHM(in_data, thres_auto[marker], FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker=marker)
			window.mainloop()
			
			#Save threshold
			thres_auto[marker] = intens_thres
			
		#Filter cells for autofluorescence
		filter_data['autofluo_w2_pass'] = filter_data['autofluo_cycle1_w2'] < thres_auto['autofluo_cycle1_w2']
		filter_data['autofluo_w3_pass'] = filter_data['autofluo_cycle1_w3'] < thres_auto['autofluo_cycle1_w3']
		filter_data['autofluo_w4_pass'] = filter_data['autofluo_cycle1_w4'] < thres_auto['autofluo_cycle1_w4']
		
		filter_data = filter_data.loc[(filter_data['autofluo_w2_pass'] == True) & (filter_data['autofluo_w3_pass'] == True) & (filter_data['autofluo_w4_pass'] == True), ]        
		
		#Create marker lists for the different types of thresholding
		loc_markers_exp = [marker for marker in markersA if marker in markers_exp]
		loc_markers_FWHM = [marker for marker in markersA if marker in markers_FWHM]
		
		#Perform combined thresholding
		thres_intens, thres_skew, thres_ent, exp_params1, exp_params2 = combined_thresholding(marker_dict, 
																						  markers_use = loc_markers_exp, 
																						  data = filter_data,
																						  skewness_factors=skewness_factors,
																						  aberrant_scale=aberrant_scale,
																						  measurement_x = intensity_measure, 
																						  measurement_y = 'prop_50', 
																						  plot_dir = data_dir+'data_processed/diagnostics/'+sam.split('/')[-1], 
																						  der_mid=0.1,
																						  graphics_save=True, 
																						  graphics_show=True)
		
		#Perform manual check with GUI
		for marker in loc_markers_exp:
			
			#Prepare data to create GUI object
			in_data = filter_data[['FOV', 'label', 'unique_ident', '_'.join([intensity_measure, marker_dict[marker]]), '_'.join(['skewness', marker_dict[marker]]), '_'.join(['prop_50', marker_dict[marker]])]]
			in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity', 'skewness', 'prop50']

			FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
			FOV_ranks = FOV_ranks.index.to_list()
			
			basename_qsub = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/qsuboptred_'+patient+'_'+marker_dict[marker]+'_'
			basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
			FOV_dict = {FOV: basename_qsub+FOV+'.TIF' for FOV in FOV_ranks}
			seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
			#GUI
			window = GUI_exp(in_data, thres_intens[marker], thres_skew[marker], thres_ent[marker], exp_params2[marker], prop_thres_strictness = 'high',
							 FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker=marker)
			window.mainloop()

			#Save thresholds
			thres_intens[marker] = intens_thres
			thres_skew[marker] = skew_thres
			thres_ent[marker] = ent_thres

		#Perform FWHM thresholding
		thres_FWHM = FWHM_thresholding(data = filter_data, 
									   markers_FWHM = loc_markers_FWHM, 
									   base_scale_factor=1, 
									   base_symmetry='right')

		#Perform manual check with GUI
		for marker in loc_markers_FWHM:
			
			#Prepare data to create GUI object
			in_data = filter_data[['FOV', 'label', 'unique_ident', '_'.join([intensity_measure, marker_dict[marker]]), '_'.join(['skewness', marker_dict[marker]]), '_'.join(['prop_50', marker_dict[marker]])]]
			in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity', 'skewness', 'prop50']
			
			FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
			FOV_ranks = FOV_ranks.index.to_list()
			
			basename_qsub = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/qsuboptred_'+patient+'_'+marker_dict[marker]+'_'
			basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
			FOV_dict = {FOV: basename_qsub+FOV+'.TIF' for FOV in FOV_ranks}
			seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
			#GUI
			window = GUI_FWHM(in_data, thres_FWHM[marker], FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker=marker)
			window.mainloop()
			
			#Save thresholds
			thres_FWHM[marker] = intens_thres
		
		#Create and save dictionary containing model parameters
		model_params = {'thres_intens': thres_intens, 'thres_skew': thres_skew, 'thres_ent': thres_ent, 'exp_params1': exp_params1, 'exp_params2': exp_params2, 'thres_FWHM': thres_FWHM}
		
		txtfilename = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/'+name_params
		with open(txtfilename, 'wb') as file:
			pickle.dump(model_params, file)
			
		#Save intermediary output csv file
		filter_data.to_csv(data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/Data_postQC.csv', index=False)
			
			
def pheno_part1_assign(out_file_name='tmp_Phenotype_data_postTumorSplit.csv', name_params='Tumor_split_model_params.txt'):

	"""
	Assigns KI67 status to cells and separates them into tumor versus immune categories. 
	This function uses the pre-processed and quality-controlled data, utilizing the existing thresholds and model parameters 
	from a pheno_part1() to classify cells based on their marker expression profiles. It updates the dataset 
	with new columns indicating each cell's raw marker combination, KI67 status, and a final 
	assignment of the cell's origin (tumor, immune, undefined, or unresolvable conflict). The updated dataset is 
	saved as an intermediary CSV file for downstream analysis.

	Parameters
	----------
	out_file_name : str, optional
		The name of the output CSV file where the post-separation data will be saved. 
		Default is 'tmp_Phenotype_data_postTumorSplit.csv'.
	name_params : str, optional
		The filename for loading the model parameters used for cell classification. 
		Default is 'Tumor_split_model_params.txt'.

	Returns
	-------
	None : This function does not return a value. It saves the updated dataset with KI67 status and 
	tumor vs. non-tumor separation information directly to the filesystem.
	"""
	
	#Iterate through samples
	for sam in sample_list:

		#Status message
		print('Currently performing tumor-non-tumor split for ', sam.split('/')[-1])
		
		#Read post QC data
		input_path = data_dir + 'data_processed/temp_data/' + sam.split('/')[-1] + '/Data_postQC.csv'
		filter_data = pd.read_csv(input_path)
	   
		#Load existing thresholds
		thres_filename = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/'+name_params
		with open(thres_filename, 'rb') as file:
			model_params = pickle.load(file)
			
		#Assign raw identities
		raw_ident_list = raw_idents(data = filter_data, 
									marker_dict = {k:m for k,m in marker_dict.items() if k in markersA}, 
									thres_intens = model_params['thres_intens'], 
									thres_skew = model_params['thres_skew'], 
									thres_ent = model_params['thres_ent'], 
									exp_params = model_params['exp_params2'], 
									thres_FWHM = model_params['thres_FWHM'], 
									prop_thres_strictness='high') 
		
		#Add to dataframe as new column
		filter_data['raw_marker_comb1_tumor'] = raw_ident_list
		
		#Assign KI67 status in additional column
		filter_data['KI67-status'] = list(map(KI67, filter_data['raw_marker_comb1_tumor']))
		
		#Perform tumor vs. non-tumor split

		#Initialize list to store identities
		ident_list = []

		#Process each cell individually
		for cell in filter_data['unique_ident']:
		
			raw_combi = filter_data.loc[filter_data['unique_ident'] == cell, 'raw_marker_comb1_tumor'].to_list()[0]            
			new_id = ''
			
			combi_set = set(raw_combi.split('_'))
			combi_set = combi_set - {'KI67'}

			#Handle undefined case
			if combi_set == {'undefined'} or combi_set == {}:
				
				new_id = 'undefined'
				ident_list.append(new_id)
				continue

			#Handle tumor vs. non-tumor
			else:
				
				#Assign p63+ cells 'Tumor' identity (high confidence nuclear marker)
				if 'p63' in combi_set:
					new_id = 'Tumor'
					ident_list.append(new_id)
					continue
				
				#Assign cells that are exclusively CD45+ 'Immune' identity
				elif combi_set == {'CD45'}:
					new_id = 'Immune'
					ident_list.append(new_id)
					continue
				
				#Assign cells with no CD45 but at least one tumor marker 'Tumor' identity    
				elif 'CD45' not in combi_set:
					new_id = 'Tumor'
					ident_list.append(new_id)
					continue
				
				#Handle cells that are both positive for CD45 and at least one tumor marker 
				else:

					#Get the cell's CD45 percentile
					pos_data = np.array(filter_data.loc[filter_data['raw_marker_comb1_tumor'].str.contains('CD45'), '_'.join([intensity_measure, marker_dict['CD45']])])
					cell_data = filter_data.loc[filter_data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict['CD45']])].to_list()[0]
					percentile_CD45 = (pos_data < cell_data).mean()
					
					#Get the cell's mean percentile of the remanining positive tumor markers (exluding p63)
					rem_set = combi_set - {'CD45'}
					percentiles = []
					
					for m in rem_set:
						pos_data = np.array(filter_data.loc[filter_data['raw_marker_comb1_tumor'].str.contains(m), '_'.join([intensity_measure, marker_dict[m]])])
						cell_data = filter_data.loc[filter_data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict[m]])].to_list()[0]
						percentile = (pos_data < cell_data).mean()
						percentiles.append(percentile)
						
					percentile_tumor = np.mean(percentiles)

					#Assign identity based on the higher percentile
					if percentile_CD45 > percentile_tumor:
						new_id = 'Immune'
						ident_list.append(new_id)
						continue

					elif percentile_CD45 < percentile_tumor:
						new_id = 'Tumor'
						ident_list.append(new_id)
						continue

					else:
						new_id = 'unresolvable_conflict'
						ident_list.append(new_id)

		#Add cell origin (Tumor/Immune) to dataframe as new column
		filter_data['Origin'] = ident_list
		
		#Save intermediary output csv file
		filter_data.to_csv(data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/'+out_file_name, index=False)
		
		
def pheno_partX(markersX, in_file_name = 'tmp_Phenotype_data_postTumorSplit.csv', step=1):
	
	"""
	Does thresholding for selected immune markers to perform downstream immune cell classifications. 
	Focuses on cells previously identified as 'Immune' or 'undefined' and uses either FWHM or combined / negative exponential
	thresholding depending on the marker. Generates and saves raw marker combinations, 
	thresholds, and parameters for downstream analysis steps.

	Parameters
	----------
	markersX : list
		The list of markers to be thresholded in this step.
	in_file_name : str, optional
		The input CSV file name containing the cellular data, with column indicating cell origin.
		Default is 'tmp_Phenotype_data_postTumorSplit.csv'.
	step : int, optional
		The marker batch number in the thresholding process. Defaults to 1.

	Returns
	-------
	None : This function does not return a value. It updates the dataset with new raw marker combinations and 
	saves it along with the thresholding parameters to the filesystem for downstream analysis.
	"""
	
	out_file_name='/tmp_Phenotype_dataV'+str(step)+'.csv'
	out_col_name='raw_marker_comb'+str(step)
	out_thresholdfile_name='/model_params_step'+str(step)+'.txt'
	
	#Iterate through samples
	for sam in sample_list:

		#Status message
		print('Currently thresholding selected markers for ', sam.split('/')[-1])
		
		patient = sam.split('/')[-1]
		patient = patient.split('_')[0]
		
		#Read data
		input_path = data_dir + 'data_processed/temp_data/' + sam.split('/')[-1] + '/' + in_file_name
		input_data = pd.read_csv(input_path)
		
		#Select only immune cells and undefined cells
		filter_data = input_data.loc[(input_data['Origin'] == 'Immune') | (input_data['Origin'] == 'undefined'), ]
		
		#Create marker lists
		loc_markers_exp = [marker for marker in markersX if marker in markers_exp]
		loc_markers_FWHM = [marker for marker in markersX if marker in markers_FWHM]
		
		#Perform combined thresholding
		thres_intens, thres_skew, thres_ent, exp_params1, exp_params2 = combined_thresholding(marker_dict, 
																						  markers_use = loc_markers_exp, 
																						  data = filter_data,
																						  skewness_factors=skewness_factors,
																						  aberrant_scale=aberrant_scale,
																						  measurement_x = intensity_measure, 
																						  measurement_y = 'prop_50', 
																						  plot_dir = data_dir+'data_processed/diagnostics/'+sam.split('/')[-1], 
																						  der_mid=0.1,
																						  graphics_save=True, 
																						  graphics_show=True)
		
		#Perform manual check with GUI
		for marker in loc_markers_exp:
			
			#Prepare data to create GUI object
			in_data = filter_data[['FOV', 'label', 'unique_ident', '_'.join([intensity_measure, marker_dict[marker]]), '_'.join(['skewness', marker_dict[marker]]), '_'.join(['prop_50', marker_dict[marker]])]]
			in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity', 'skewness', 'prop50']

			FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
			FOV_ranks = FOV_ranks.index.to_list()
			
			basename_qsub = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/qsuboptred_'+patient+'_'+marker_dict[marker]+'_'
			basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
			FOV_dict = {FOV: basename_qsub+FOV+'.TIF' for FOV in FOV_ranks}
			seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
			#GUI
			window = GUI_exp(in_data, thres_intens[marker], thres_skew[marker], thres_ent[marker], exp_params2[marker], prop_thres_strictness = 'high',
							 FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker=marker)
			window.mainloop()
			
			#Save thresholds
			thres_intens[marker] = intens_thres
			thres_skew[marker] = skew_thres
			thres_ent[marker] = ent_thres
		
		#Perform FWHM thresholding
		thres_FWHM = FWHM_thresholding(data = filter_data, 
									   markers_FWHM = loc_markers_FWHM, 
									   base_scale_factor=1.2, 
									   base_symmetry='right')

		#Perform manual check with GUI
		for marker in loc_markers_FWHM:
			
			#Prepare data to create GUI object
			in_data = filter_data[['FOV', 'label', 'unique_ident', '_'.join([intensity_measure, marker_dict[marker]]), '_'.join(['skewness', marker_dict[marker]]), '_'.join(['prop_50', marker_dict[marker]])]]
			in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity', 'skewness', 'prop50']
			
			FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
			FOV_ranks = FOV_ranks.index.to_list()
			
			basename_qsub = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/qsuboptred_'+patient+'_'+marker_dict[marker]+'_'
			basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
			FOV_dict = {FOV: basename_qsub+FOV+'.TIF' for FOV in FOV_ranks}
			seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
			#GUI
			window = GUI_FWHM(in_data, thres_FWHM[marker], FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker=marker)
			window.mainloop()
			
			#Save thresholds
			thres_FWHM[marker] = intens_thres
			
		#Create and save dictionary containing model parameters
		model_params = {'thres_intens': thres_intens, 'thres_skew': thres_skew, 'thres_ent': thres_ent, 'exp_params1': exp_params1, 'exp_params2': exp_params2, 'thres_FWHM': thres_FWHM}
		
		txtfilename = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+out_thresholdfile_name
		with open(txtfilename, 'wb') as file:
			pickle.dump(model_params, file)

		#Round X of raw identity creation
		raw_ident_list = raw_idents(data = filter_data, 
									marker_dict = {k:m for k,m in marker_dict.items() if k in markersX}, 
									thres_intens = thres_intens, 
									thres_skew = thres_skew, 
									thres_ent = thres_ent, 
									exp_params = exp_params2, 
									thres_FWHM = thres_FWHM, 
									prop_thres_strictness='high') 

		#Add raw identities to dataframe as new column
		filter_data[out_col_name] = raw_ident_list

		#Save intermediary output csv file
		filter_data.to_csv(data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+out_file_name)
			
			
def pheno_final(old_file_name = 'tmp_Phenotype_data_postTumorSplit.csv', id_cols=[1,2]):
	
	"""
	Assigns final phenotypes to non-tumor cells based on combined raw ID columns from previous steps. This function aggregates
	marker combinations to assign a comprehensive phenotype to each cell in the non-tumor dataset while resolving ID conflicts. 
	It also assigns PD1 and CTLA4 status and updates the full dataset (including tumor cells) with these final phenotypes.

	Parameters
	----------
	old_file_name : str, optional
		The filename of the dataset before final phenotype assignment. 
		Default is 'tmp_Phenotype_data_postTumorSplit.csv'.
	id_cols : list of int, optional
		The indices of the thresholding files (created with pheno_partX()) for raw ID combination and final phenotype assignment. 
		Default is [1,2].

	Returns
	-------
	None : This function does not return a value. It saves the dataset with assigned final phenotypes, including
	PD1 and CTLA4 status, to the filesystem for downstream analysis.
	"""
	
	#Iterate through samples
	for sam in sample_list:

		#Status message
		print('Currently assigning phenotypes for ', sam.split('/')[-1])
		
		#Read first data file
		input_path = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/tmp_Phenotype_dataV'+str(id_cols[0])+'.csv'
		filter_data = pd.read_csv(input_path)
		
		#Add remaining files and combine to complete dataset
		if len(id_cols) > 1:
			for step in id_cols[1:]:
				
				#Read data of next thresholding step
				input_path = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/tmp_Phenotype_dataV'+str(step)+'.csv'
				input_data = pd.read_csv(input_path)
				
				#Check for matching IDs
				if all(input_data['unique_ident'] == filter_data['unique_ident']):
					filter_data['raw_marker_comb'+str(step)] = input_data['raw_marker_comb'+str(step)]
				else:    
					print('Warning! Dataframes of seperate thresholding runs do not match! Aborting.')
					return #Exit function
			
			#Merge raw marker combinations
			filter_data['all_markers'] = filter_data[['raw_marker_comb'+str(x) for x in id_cols]].agg('_'.join, axis=1)
		
		else:
			filter_data['all_markers'] = filter_data['raw_marker_comb'+str(id_cols[0])]
			
		#Phenotyping   

		#Initialize outputs
		ident_list = []
		PD1_list = []
		CTLA4_list = []
		resolve_list = {}

		#Assign phenotype for each cell individually
		for cell in filter_data['unique_ident']:

			#Get raw marker combination
			raw_combi = filter_data.loc[filter_data['unique_ident'] == cell, 'all_markers'].to_list()[0]
			new_id = ''
			combi_set = set(raw_combi.split('_'))

			#Assign PD1 status
			if 'PD1' in combi_set:
				PD1_list.append('PD1pos')
				combi_set = combi_set - {'PD1'}
			else:
				PD1_list.append('PD1neg') 
			 
			#Assign CLTA4 status   
			if 'CTLA4' in combi_set:
				CTLA4_list.append('CTLA4pos')
				combi_set = combi_set - {'CTLA4'}
			else:
				CTLA4_list.append('CTLA4neg')

			if filter_data.loc[filter_data['unique_ident'] == cell, 'Origin'].to_list()[0] == "Immune":
				combi_set.add('CD45')
			
			#Handle undefined case
			if combi_set == {'undefined'}:
				new_id = 'undefined'
				ident_list.append(new_id)
				continue

			#Handle all other combinations
			else:
				
				combi_set = combi_set - {'undefined'}
				ident_sets = {}
				
				#Create dictionary with all possible IDs and their markers combinations for a given cell
				for id, combis in pheno_dict.items():
					combis.sort(key=len, reverse=True)
					for i in combis:
						if i.issubset(combi_set):
							ident_sets[id] = i
							break

				#Case 1: No assignable ID
				if len(list(ident_sets.keys())) == 0: 
					new_id = 'undefined_underassign'
					ident_list.append(new_id)
					continue
				
				#Case 2: Unique ID
				elif len(list(ident_sets.keys())) == 1:
					new_id = str(list(ident_sets.keys())[0])
					ident_list.append(new_id)
					continue

				#Case 3: > 1 possible ID
				else:

					conflicts = {}

					#Sort identities by number of matching markers
					sorted_ident_list = [k for k in sorted(ident_sets, key=lambda k: len(ident_sets[k]), reverse=True)]

					#Remove redundant (included) IDs
					new_keys = []
					for id in sorted_ident_list:
						bool_val = True
						
						for key in new_keys:
							
							if ident_sets[id].issubset(ident_sets[key]) == False:
								pass
							else:
								bool_val = False
								break
						
						if bool_val:
							new_keys.append(id)

					#Reduce ident_sets dictionary
					ident_sets = {k: v for k, v in ident_sets.items() if k in new_keys}
					
					#Case 1: Unique ID
					if len(list(ident_sets.keys())) == 1:
						new_id = str(list(ident_sets.keys())[0])
						ident_list.append(new_id)
						continue
					
					#Case 2: > 4 conflicting IDs --> unresolvable confict
					if len(list(ident_sets.keys())) > 4:
						new_id = 'unresolvable_conflict'
						ident_list.append(new_id)
						continue                                            

					#Case 3: Between 2 and 4 possible IDs --> perform conflict resolution

					#Initialize scores for each identity
					ident_scores = {id: 0 for id in ident_sets.keys()}

					#Iterate through priority groups
					for group in np.unique(list(pheno_markers.values())):

						#Extract markers of current priority group that the cell is positive for
						group_markers = [gm for gm, ind in pheno_markers.items() if ind == group and gm in combi_set]
						
						#Case 1: None
						if len(group_markers) == 0:
							continue
						
						#Case 2: > 0 markers  

						#Extract percentiles for each marker
						percentiles = {}
						for m in group_markers:
							pos_data = np.array(filter_data.loc[filter_data['all_markers'].str.contains(m), '_'.join([intensity_measure, marker_dict[m]])])
							cell_data = filter_data.loc[filter_data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict[m]])].to_list()[0]
							percentile = (pos_data < cell_data).mean()
							percentiles[m] = percentile
						  
						checkID = [] 

						#Calculate and update identity scores  

						#Iterate through possible IDs                     
						for id in ident_sets.keys():
							
							#In each iteration, collect the IDs that haven't been compared to current one yet (rem_ids)
							checkID.append(id)
							rem_ids = [z for z in ident_sets.keys() if z not in checkID]
							
							#Compare each ID in rem_ids with current one
							for i in rem_ids:

								#Extract mean percentiles of non-overlapping markers for both IDs
								ID_A = ident_sets[id] - ident_sets[i]
								ID_B = ident_sets[i] - ident_sets[id]
								meanA = np.mean([num for m, num in percentiles.items() if m in ID_A])
								meanB = np.mean([num for m, num in percentiles.items() if m in ID_B])
								
								#Update score for winning ID
								if meanA > meanB:
									ident_scores[id] = ident_scores[id] + 1
								elif meanA < meanB:
									ident_scores[i] = ident_scores[i] + 1
								else:
									pass
						
						#After performing all possible comparisons, extract highest scoring ID(s)     
						winning_IDs = [key for key, value in ident_scores.items() if value == max(ident_scores.values())]
						
						#Case 1: Single winning ID
						if len(winning_IDs) == 1:
							new_id = str(winning_IDs[0])
							ident_list.append(new_id)
							break

						#Case 2: Multiple winning IDs
						else:
							continue
							
					#If no ID assigned -> Assign unresolvable conflict
					if len(new_id) == 0:
						new_id = 'unresolvable_conflict'
						ident_list.append(new_id)
		
		#Save results to new columns
		filter_data['identity'] = ident_list
		filter_data['PD1-status'] = PD1_list
		filter_data['CTLA4-status'] = CTLA4_list
		
		#Re-combine data with tumor cells to create full output

		orig_path = data_dir + 'data_processed/temp_data/' + sam.split('/')[-1] + '/' + old_file_name
		orig_data = pd.read_csv(orig_path)
		
		final_idents = []
		all_markers = []
		PD1_status = []
		CTLA4_status = []

		for cell in orig_data['unique_ident']:
			
			if cell in filter_data['unique_ident'].to_list():
				final_idents.append(filter_data.loc[filter_data['unique_ident'] == cell, 'identity'].to_list()[0])
				all_markers.append(filter_data.loc[filter_data['unique_ident'] == cell, 'all_markers'].to_list()[0])
				PD1_status.append(filter_data.loc[filter_data['unique_ident'] == cell, 'PD1-status'].to_list()[0])
				CTLA4_status.append(filter_data.loc[filter_data['unique_ident'] == cell, 'CTLA4-status'].to_list()[0])
			
			else:
				final_idents.append('Tumor')
				all_markers.append('none')
				PD1_status.append('not_assessed')
				CTLA4_status.append('not_assessed')
		
		orig_data['identity'] = final_idents
		orig_data['all_markers'] = all_markers
		orig_data['PD1-status'] = PD1_status
		orig_data['CTLA4-status'] = CTLA4_status
		
		pheno_cols = ['unique_ident', 'label', 'FOV', 'centroid_x', 'centroid_y', 'area', 'raw_marker_comb1_tumor', 'all_markers', 'Origin', 'identity', 'KI67-status', 'PD1-status', 'CTLA4-status']
		phenotype_data = orig_data[pheno_cols]
		
		#Save reduced and full versions of output dataframe
		phenotype_data.to_csv(data_dir+'data_processed/derivative_data/'+ sam.split('/')[-1]+'/Phenotype_data.csv')  
		filter_data.to_csv(data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/tmp_Phenotype_dataFinal_allData.csv')
		
		
def subset_markers():
	
	"""
	Performs thresholding to assign TCF1, FoxP3, and CD163 status in their respective cell subsets within the whole dataset.

	Parameters
	----------
	None

	Returns
	-------
	None : This function does not return a value. It updates the dataset with TCF1, FoxP3, and CD163 statuses for 
	relevant cell subsets and saves this updated dataset for downstream analysis.
	"""
	
	#Iterate through samples
	for sam in sample_list:

		#Status message
		print('Currently checking subset markers for ', sam.split('/')[-1])
		
		patient = sam.split('/')[-1]
		patient = patient.split('_')[0]
		
		#Read data
		input_path = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/tmp_Phenotype_dataFinal_allData.csv'
		filter_data = pd.read_csv(input_path)
		
		pheno_path = data_dir+'data_processed/derivative_data/'+ sam.split('/')[-1]+'/Phenotype_data.csv'
		pheno_data = pd.read_csv(pheno_path)
		
		#Reduce data to create myeloid and T cell dataset
		Myeloid_data = filter_data.loc[filter_data['identity'].str.contains('Neutrophil|Macrophage|Monocyte|myeloid', regex=True), ]
		Tcell_data = filter_data.loc[filter_data['identity'].str.contains('Tcell'), ]
		
		#Perform combined thresholding for CD163 using myeloid data
		thres_intens, thres_skew, thres_ent, exp_params1, exp_params2 = combined_thresholding(marker_dict, 
																						  markers_use = ['CD163'], 
																						  data = Myeloid_data,
																						  skewness_factors=skewness_factors,
																						  aberrant_scale=aberrant_scale,
																						  measurement_x = intensity_measure, 
																						  measurement_y = 'prop_50', 
																						  plot_dir = data_dir+'data_processed/diagnostics/'+sam.split('/')[-1], 
																						  der_mid=0.1,
																						  graphics_save=True, 
																						  graphics_show=True)
		
		#Perform FWHM thresholding for TCF1 and FoxP3 using T cell data
		thres_FWHM = FWHM_thresholding(data = Tcell_data, 
									   markers_FWHM = ['TCF1', 'FoxP3'], 
									   base_scale_factor=1.2, 
									   base_symmetry='right')
		
		#Perform manual check with GUI for CD163

		#Prepare data to create GUI object
		in_data = Myeloid_data[['FOV', 'label', 'unique_ident', '_'.join([intensity_measure, marker_dict['CD163']]), '_'.join(['skewness', marker_dict['CD163']]), '_'.join(['prop_50', marker_dict['CD163']])]]
		in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity', 'skewness', 'prop50']

		FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
		FOV_ranks = FOV_ranks.index.to_list()
			
		basename_qsub = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/qsuboptred_'+patient+'_'+marker_dict['CD163']+'_'
		basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
		FOV_dict = {FOV: basename_qsub+FOV+'.TIF' for FOV in FOV_ranks}
		seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
		#GUI
		window = GUI_exp(in_data, thres_intens['CD163'], thres_skew['CD163'], thres_ent['CD163'], exp_params2['CD163'], prop_thres_strictness = 'high',
						 FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker='CD163')
		window.mainloop()
			
		#Save threshold
		thres_intens['CD163'] = intens_thres
		thres_skew['CD163'] = skew_thres
		thres_ent['CD163'] = ent_thres
		
		#Perform manual check with GUI for TCF1 and FoxP3
		for marker in ['TCF1', 'FoxP3']:

			#Prepare data to create GUI object
			in_data = Tcell_data[['FOV', 'label', 'unique_ident', '_'.join([intensity_measure, marker_dict[marker]]), '_'.join(['skewness', marker_dict[marker]]), '_'.join(['prop_50', marker_dict[marker]])]]
			in_data.columns = ['FOV', 'label', 'unique_ident', 'intensity', 'skewness', 'prop50']
			
			FOV_ranks = in_data.groupby('FOV').sum().sort_values(by='intensity', ascending=False)
			FOV_ranks = FOV_ranks.index.to_list()
			
			basename_qsub = data_dir+'data_processed/data_crop_qsub_red/'+sam.split('/')[-1]+'/qsuboptred_'+patient+'_'+marker_dict[marker]+'_'
			basename_seg = data_dir+'data_processed/data_seg_masks_red/'+sam.split('/')[-1]+'/maskcell_'
			FOV_dict = {FOV: basename_qsub+FOV+'.TIF' for FOV in FOV_ranks}
			seg_dict = {FOV: basename_seg+FOV+'.TIF' for FOV in FOV_ranks}
			
			#GUI
			window = GUI_FWHM(in_data, thres_FWHM[marker], FOV_ranks=FOV_ranks, FOV_dict=FOV_dict, seg_dict=seg_dict, marker=marker)
			window.mainloop()
			
			#save thresholds
			thres_FWHM[marker] = intens_thres
			
		#Create and save dictionary containing model parameters
		model_params = {'thres_intens': thres_intens, 'thres_skew': thres_skew, 'thres_ent': thres_ent, 'exp_params1': exp_params1, 'exp_params2': exp_params2, 'thres_FWHM': thres_FWHM}
		
		txtfilename = data_dir+'data_processed/temp_data/'+ sam.split('/')[-1]+'/model_params_subsetting.txt'
		with open(txtfilename, 'wb') as file:
			pickle.dump(model_params, file)
		
		#Initialize outputs
		FoxP3_status = []
		TCF1_status = []
		CD163_status = []
		
		#Extract myeloid and T cell IDs
		Myeloid_ID = Myeloid_data['unique_ident'].to_list()
		Tcell_ID = Tcell_data['unique_ident'].to_list()
		
		#Create scaled version of myeloid data (for combined thresholding status assignment)
		scale_data_dict = dict(zip(Myeloid_data['unique_ident'], scale(Myeloid_data['_'.join([intensity_measure, marker_dict['CD163']])])))
		
		#Subtype each cell individually
		for cell in pheno_data['unique_ident']:
			
			#T cells
			if cell in Tcell_ID:
				
				CD163_status.append('not_assessed')
				
				#Assign TCF1 status
				if Tcell_data.loc[Tcell_data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict['TCF1']])].to_list()[0] > thres_FWHM['TCF1']:
					TCF1_status.append('TCF1pos')
				else:
					TCF1_status.append('TCF1neg')
					
				#Assign FoxP3 status
				if Tcell_data.loc[Tcell_data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict['FoxP3']])].to_list()[0] > thres_FWHM['FoxP3']:
					FoxP3_status.append('FoxP3pos')
				else:
					FoxP3_status.append('FoxP3neg')
			
			#Myeloid cells
			elif cell in Myeloid_ID:
				
				TCF1_status.append('not_assessed')
				FoxP3_status.append('not_assessed')
				
				#Collect data
				intens = Myeloid_data.loc[Myeloid_data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict['CD163']])].to_list()[0]
				prop = Myeloid_data.loc[Myeloid_data['unique_ident'] == cell, '_'.join(['prop_50', marker_dict['CD163']])].to_list()[0]
				skew = Myeloid_data.loc[Myeloid_data['unique_ident'] == cell, '_'.join(['skewness', marker_dict['CD163']])].to_list()[0]

				#Prepare dynamic prop50 threshold (strictness: high)

				prop_thres_width = thres_intens['CD163'][1] - thres_intens['CD163'][0]

				if (intens - thres_intens['CD163'][1]) < 0:
					prop_thres_diff = thres_ent['CD163'][0] - thres_ent['CD163'][1]
					prop50 = ((intens - thres_intens['CD163'][0]) / prop_thres_width) * 0.5 * prop_thres_diff + scale(i = negative_exponential(scale_data_dict[cell], *exp_params2['CD163']), reverse=True, raw=Myeloid_data['_'.join(['prop_50', marker_dict['CD163']])]) + 0.5 * prop_thres_diff
				else:
					prop50 = 1

				#Assign CD163 status
				if (prop < prop50) & (intens > thres_intens['CD163'][0]) & (skew < thres_skew['CD163']):
					CD163_status.append('CD163pos')
				else:
					CD163_status.append('CD163neg')

			#Other cells
			else:
				TCF1_status.append('not_assessed')
				FoxP3_status.append('not_assessed')
				CD163_status.append('not_assessed')
		
		#Add results to dataframe   
		pheno_data['CD163-status'] = CD163_status
		pheno_data['TCF1-status'] = TCF1_status
		pheno_data['FoxP3-status'] = FoxP3_status
		
		#Save final phenotyped and subtyped data
		pheno_data.to_csv(data_dir+'data_processed/derivative_data/'+ sam.split('/')[-1]+'/Phenotype_data_subset.csv')
	

def prelim_graphics():
	
	"""
	Generates and saves preliminary graphical representations of phenotype distributions across samples. 
	This function creates phenotype maps for each field of view (FOV) within each sample, highlighting 
	the distribution of identified cell phenotypes. These maps are saved as one-hot encoded masks with 
	specific phenotypes highlighted for visual inspection and preliminary analysis.

	Parameters
	----------
	None

	Returns
	-------
	None : This function does not return a value. It directly saves phenotype distribution maps as images 
	in a specified directory structure, facilitating quick visual assessment of phenotype distributions 
	across different samples.
	"""
	
	#Iterate through samples
	for sam in sample_list:

		#Status message
		print('Currently creating graphics for ', sam.split('/')[-1])
		
		#Read data
		input_path = data_dir+'data_processed_reprocess/derivative_data/'+ sam.split('/')[-1]+'/Phenotype_data.csv'
		filter_data = pd.read_csv(input_path)
		
		#Make output directory
		if os.path.isdir(data_dir+'data_processed_reprocess/phenotype_masks/'+ sam.split('/')[-1]) == False:
			os.makedirs(data_dir+'data_processed_reprocess/phenotype_masks/'+ sam.split('/')[-1])
		
		#Initialize
		filter_data['num_ident'] = np.ones((len(filter_data)), dtype='int32')
		ids = list(pheno_dict.keys())
		ids.append('Tumor')

		#Iterate through FOVs
		for FOV in set(filter_data['FOV']):
			
			#Reduce data
			temp_data = filter_data.loc[filter_data['FOV'] == FOV, : ]

			#Load cell mask
			maskpath = data_dir+'data_processed/data_seg_masks_red/'+ sam.split('/')[-1] + '/maskcell_' + FOV + '.TIF'
			temp_mask = imread(maskpath, plugin='pil')

			#Create and save one-hot encoded masks for each phenotype (phenotype of interest in yellow)
			for ident in ids: 

				temp_data.loc[temp_data['identity'] == ident, 'num_ident'] = 2
				one_hot_hm = mask_heatmap(temp_mask, temp_data, 'num_ident')
				Hm_rgb = label2rgb(one_hot_hm, colors = ['black', 'grey', 'yellow'])

				save_path = data_dir+'data_processed_reprocess/phenotype_masks/'+ sam.split('/')[-1] + '/Phenotype_mask_' + ident + '_' + FOV + '.png'
				Img.fromarray(np.uint8(Hm_rgb*255)).save(save_path)

				temp_data['num_ident'] = np.ones((len(temp_data)), dtype='int32')



##PROCESSING

prep_plotting()
pheno_part1(name_params='Tumor_split_model_params.txt')
pheno_part1_assign(out_file_name='tmp_Phenotype_data_postTumorSplit.csv', name_params='Tumor_split_model_params.txt')
pheno_partX(markersX=markersJan, in_file_name = 'tmp_Phenotype_data_postTumorSplit.csv', step=1)
pheno_partX(markersX=markersJulie, in_file_name = 'tmp_Phenotype_data_postTumorSplit.csv', step=2)
pheno_final(old_file_name = 'tmp_Phenotype_data_postTumorSplit.csv', id_cols=[1,2])
subset_markers()
prelim_graphics()