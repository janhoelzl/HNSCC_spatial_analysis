'''
Functions for part 3 of main HNSCC image analysis pipeline: Cell phenotyping
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''



###FUNCTIONS

def mask_heatmap(mask, props, numID):
	
	"""
	Maps properties from a DataFrame to a labeled mask, assigning each unique label in the mask
	a value based on a specified property. Slightly adapted version of the identically named function in part 2 that also handles removed cells.

	Parameters
	----------
	mask : array
		A 2D numpy array representing a labeled cell mask, where background is assumed to be 0 and each
		unique label corresponds to a different cell (or more general: region).
	props : DataFrame
		A pandas DataFrame containing properties for each cell. Must include a 'label'
		column that matches labels in the mask and a column for the specified identity to map.
	numID : str
		The column name in props DataFrame whose values are to be mapped to the corresponding labels
		in the mask.

	Returns
	-------
	int_mask : array
		A numpy array of the same shape as the cell mask, where each label in the original mask is replaced
		with the corresponding value from the 'identity' column in props.
	"""

	assert numID in props.columns

	int_mask = mask.copy()

	for c in np.unique(int_mask.flatten())[1:]:
		
		if c in props['label'].to_list():
			int_mask[np.ma.masked_where(mask==c,mask).mask] = props[props['label']==c][numID].values[0]
		else:
			int_mask[np.ma.masked_where(mask==c,mask).mask] = 0
			
	return int_mask


def annotate(mask, dotsize):
	
	"""
	Annotates a labeled mask by drawing a circle of specified radius at the centroid of each labeled cell/region.

	Parameters
	----------
	mask : array
		Cell mask (background 0s, cells with unique identifiers)
	dotsize : int
		The radius of the circles to be drawn in pixels.

	Returns
	-------
	int_mask : array
		A numpy array of the same shape as the input mask, where a circle of radius `dotsize` is drawn at the
		centroid of each labeled region. The circle's label matches the label of the region.

	Notes
	-----
	- If a cell's area is smaller than the circle to be drawn, the circle may extend beyond it's original boundaries.
	"""

	#Prepare output and list of unique cells
	int_mask = mask.copy()
	mask_unique = np.unique(int_mask.flatten())[1:]
	
	for c in mask_unique:

		#Get cell coordinates
		obj_coord = np.where(int_mask==c)
		if obj_coord[0].size != 0 and obj_coord[1].size != 0:

			#Create new entry
			obj_mask = int_mask[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1]
			y_center, x_center = np.argwhere(obj_mask==c).sum(0)/(obj_mask==c).sum()
			new_entry = np.zeros(obj_mask.shape)
			rr, cc = draw.disk((y_center, x_center), radius=dotsize, shape=new_entry.shape)
			new_entry[rr, cc] = c

			#Edit int mask
			int_mask[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1] = new_entry

	return int_mask


def FWHM(marker, data, asym='sym', scale_factor=1): 
	
	"""
	Finds the Full Width at Half Maximum (FWHM) for the highest and leftmost peak in the data using a Gaussian KDE.

	Parameters
	----------
	marker : str
		Data identifier (Marker) to be thresholded, used for output messages.
	data : array-like
		The dataset for which the peak and FWHM are to be calculated (typically some type of fluorescence intensity).
	asym : {'sym', 'left', 'right'}, optional
		Assumption about the data distribution to adjust the threshold:
		- 'sym' (default): Assumes symmetry and calculates threshold based on FWHM.
		- 'left': Assumes the threshold to be on the left side of the data distrubution and adjusts threshold accordingly.
		- 'right': Assumes the threshold to be on the right side of the data distrubution and adjusts threshold accordingly.
	scale_factor : float, optional
		Scaling factor applied to the threshold calculation, default is 1.

	Returns
	-------
	threshold : float
		The calculated threshold based on peak, FWHM, and asymmetry assumption.
	peak : float
		The x-value of the identified peak.
	width : float
		The Full Width at Half Maximum (FWHM) of the peak.
	height : float
		The height of the peak.
	left_ips : float
		The left intercept position of the FWHM.
	right_ips : float
		The right intercept position of the FWHM.

	Notes
	-----
	- The function prints the peaks found, their prominences, and a warning if a second peak more than 3-fold higher 
	  than the leftmost one is identified, in which case the higher one is used for thresholding.
	- It uses a Gaussian Kernel Density Estimate (KDE) to smooth the data and identify peaks.
	- The 'asym' parameter allows adjusting for skewed distributions where only one side is relevant by modifying the threshold calculation.
	"""
	
	#Prepare KDE
	kde = gaussian_kde(data)
	x = np.linspace(np.min(data), np.max(data), np.array(data).size)
	y = kde.evaluate(x)
	
	#Find peaks
	peaks, properties = find_peaks(y, prominence=0)
	print(f'Peaks for {marker} are: {x[peaks]}.')
	print('Prominences: ', properties["prominences"])
	
	#Identity peak of interest
	exp_peak = peaks[0]
	for peak in peaks:
		
		if kde(x[peak]) > 3*kde(x[exp_peak]):
			print("Warning: Peak 3-fold higher than previous peak of interest identified! Higher one is used.")
			exp_peak = peak
	
	#Extract necessary data about peak at 0.5* it's height (ips, width, height, ...)
	width, height, left_ips, right_ips = peak_widths(y, np.array([exp_peak]), rel_height=0.5)
	width = round(width[0]) * ((np.max(x) - np.min(x)) / x.size)
	left_ips = x[round(left_ips[0])]
	right_ips = x[round(right_ips[0])]
	print('width: ', width, 'left_ips: ', left_ips, 'right_ips: ', right_ips, 'peak: ', x[exp_peak])
	
	#Calculate thresholds based on 'asym' setting
	if asym == 'left':
		thres = x[exp_peak] + scale_factor * (2 * ((x[exp_peak]) - left_ips))
	elif asym == 'right':
		thres = x[exp_peak] + scale_factor * (2 * (right_ips - x[exp_peak]))
	else:
		thres = x[exp_peak] + scale_factor * width
	
	return thres, x[exp_peak], width, float(height), left_ips, right_ips


def logFWHM(marker, in_data, asym='sym', scale_factor=1):
	
	"""
	In principle identical to FWHM() function, but performs log2-transform before thresholding.
	Especially suited for approximately log-normally distributed data.
	"""
	
	#Log2 transform data (+1 to avoid taking the log of 0s)
	data = np.log2(in_data + 1)

	#Prepare KDE
	kde = gaussian_kde(data)
	x = np.linspace(np.min(data), np.max(data), np.array(data).size)
	y = kde.evaluate(x)
	
	#Find peaks
	peaks, properties = find_peaks(y, prominence=0.1) #prominence threshold to exclude very small peaks
	print(f'Peaks for {marker} are: {x[peaks]}.')
	print('Prominences: ', properties["prominences"])
	
	#Identity peak of interest
	exp_peak = peaks[0]
	for peak in peaks:

		if kde(x[peak]) > 3*kde(x[exp_peak]):
			print("Warning: Leftmost and highest peak are not identical! Highest is used.")
			exp_peak = peak
	
	#Extract necessary data about peak at 0.5* it's height (ips, width, height, ...)
	width, height, left_ips, right_ips = peak_widths(y, np.array([exp_peak]), rel_height=0.5)
	width = round(width[0]) * ((np.max(x) - np.min(x)) / x.size)
	left_ips = x[round(left_ips[0])]
	right_ips = x[round(right_ips[0])]
	print('width: ', width, 'left_ips: ', left_ips, 'right_ips: ', right_ips, 'peak: ', x[exp_peak])
	
	#Calculate thresholds in non-log transformed space based on 'asym' setting
	if asym == 'left':
		thres = 2 ** (x[exp_peak] + scale_factor * (2 * (x[exp_peak] - left_ips))) - 1
	elif asym == 'right':
		thres = 2 ** (x[exp_peak] + scale_factor * (2 * (right_ips - x[exp_peak]))) - 1
	else:
		thres = 2 ** (x[exp_peak] + scale_factor * width) - 1
	
	return thres, x[exp_peak], width, float(height), left_ips, right_ips

		
def scale(i, idents=None, linearize=False, reverse=False, raw=None):

	"""
	Custom scaling helper function.
	Scales an input array 'i' according to specified parameters, supporting linearization, reversal, and raw scaling.

	Parameters
	----------
	i : array-like
		The input array to be scaled.
	idents : array-like, optional
		Identifiers corresponding to each element in 'i'. Required for linearization.
	linearize : bool, optional
		If True, linearly interpolates 'i' based on 'idents', default is False.
	reverse : bool, optional
		If True, reverses the scaling operation based on 'raw', default is False.
	raw : array-like, optional
		The original array from which 'i' was scaled. Required for reversing scaling or scaling with reference.

	Returns
	-------
	scaled_out : array-like
		The scaled array.

	Notes
	-----
	- If 'linearize' is True, 'i' is linearly interpolated to a range of [0, 1] based on 'idents'.
	- If 'reverse' is True and 'raw' is provided, 'i' is scaled back to the range of 'raw'.
	- If 'reverse' is False and 'raw' is provided, 'i' is normalized based on the range of 'raw'.
	- Without 'linearize' or 'raw', 'i' is normalized to its own range, with consideration for negative values.
	"""
	
	#Linearize
	if linearize:

		if (idents == None).all():
			idents = range(0, np.array(i).size, 1)
			
		rang = list(np.linspace(0, 1, np.array(i).size))
		combined = zip(list(i), list(idents))
		sortlist = sorted(combined)
		combined_2 = zip(sortlist, rang)
		sortlist_2 = sorted(combined_2, key = lambda y: y[0][1])
		scaled_out = [element for _, element in sortlist_2]

	#Reverse scaling to raw
	elif reverse == True and raw is not None:

		scaled_out = i * (np.max(raw) - np.min(raw)) + np.min(raw)
			
	#Scaling to raw
	elif reverse == False and raw is not None:

		if np.min(raw) < 0:
			scaled_out = (i + abs(np.min(raw))) / (np.max(raw) - np.min(raw))
		elif np.min(raw) == 0:
			scaled_out = i / np.max(raw)
		else:
			scaled_out = (i - np.min(raw)) / (np.max(raw) - np.min(raw))
		 
	#Standard scaling to [0, 1]   
	else:

		if np.min(i) < 0:
			scaled_out = (i + abs(np.min(i))) / (np.max(i) - np.min(i))
		elif np.min(i) == 0:
			scaled_out = i / np.max(i)
		else:
			scaled_out = (i - np.min(i)) / (np.max(i) - np.min(i))
			
	return scaled_out


def negative_exponential(x, b, k):

	"""
	Calculates the negative exponential of input values.

	Parameters
	----------
	x : array-like
		The input values.
	b : float
		Scaling factor.
	k : float
		Exponential rate.

	Returns
	-------
	array-like
		The negative exponential of the input values.
	"""

	return b * (1 - np.exp(-k * x))


def negative_exponential_jacobian(x, b, k):

	"""
	Computes the Jacobian matrix of the negative exponential function with respect to its parameters.

	Parameters
	----------
	x : array-like
		The input values.
	b : float
		Scaling factor.
	k : float
		Exponential rate.

	Returns
	-------
	array-like
		The Jacobian matrix of the negative exponential function for each input in `x`.
	"""

	return np.array([1 - np.exp(-k * x), b * x * np.exp(-k * x)]).T


def combined_thresholding(marker_dict, markers_use, data, skewness_factors, aberrant_scale, measurement_x:str, plot_dir:str, 
						  measurement_y='prop_50', der_mid=0.1, graphics_save=True, graphics_show=True):
	
	"""
	Main thresholding function (see extended methods) based on curve fitting, skewness, and intensity rankings.
	Visualizes the results in various QC plots.

	Parameters
	----------
	marker_dict : dict
		A dictionary mapping markers to cycles/channels.
	markers_use : list
		A list of markers to be thresholded.
	data : DataFrame
		A pandas DataFrame containing the data for analysis.
	skewness_factors : dict
		A dictionary of pre-set scale factors for skewness thresholding for each marker.
	aberrant_scale : dict
		A dictionary of pre-set scale factors for aberrant scaling for each marker.
	measurement_x : str
		Fluorescence measurement type to use for thresholding (e.g. mean intensity, inter quantile mean, ...).
	plot_dir : str
		The directory where graphics will be saved.
	measurement_y : str, optional
		The prefix of the column names in `data` for the y-axis measurements. Defaults to 'prop_50' (entropy like).
	der_mid : float, optional
		The derivative midpoint for intensity thresholding. Defaults to 0.1.
	graphics_save : bool, optional
		If True, saves generated graphics to `plot_dir`. Defaults to True.
	graphics_show : bool, optional
		If True, displays the generated graphics. Defaults to True.

	Returns
	-------
	tuple
		Contains dictionaries with thresholds for intensity, skewness, and entropy/prop_50; 
		as well as experimental parameters from the first and second curve fitting.

	Notes
	-----
	- This function processes each specified marker by fitting a negative exponential curve to the scaled data, 
	  calculating intensity and skewness thresholds, and optionally visualizing the curve fitting and histograms 
	  for skewness and prop_50 thresholds.
	- It uses a two-step process for curve fitting, first on the original data and then on data filtered by skewness 
	  thresholding, in order to refine the thresholds.
	- Graphics options allow for saving and/or displaying scatter plots of the curve fits and histograms of skewness 
	  and prop_50 thresholds.
	"""
	
	#Create output dictionaries
	thres_intens = {}
	exp_params1 = {}
	exp_params2 = {}
	thres_skew = {}
	thres_ent = {}

	for marker in markers_use:
		
		#Load data
		xdata = scale(data['_'.join([measurement_x, marker_dict[marker]])])
		ydata = scale(data['_'.join([measurement_y, marker_dict[marker]])])
		xdata_raw = data['_'.join([measurement_x, marker_dict[marker]])]
		ydata_raw = data['_'.join([measurement_y, marker_dict[marker]])]

		#1st fitting (on scaled data)
		popt, pcov = curve_fit(f=negative_exponential,
							   jac=negative_exponential_jacobian,
							   xdata=xdata,
							   ydata=ydata,
							   method='lm', #Levenberg-Marquard Algorithm
							   p0=[0.9, 10]) #initial guess: 0.9 for asymptote, 10 for growth rate usually works

		exp_params1[marker] = popt
		x = np.linspace(0, 1, len(xdata))
		y = negative_exponential(x, *popt)

		#1st derivative
		dx = x[1]-x[0]
		f1 = np.gradient(y, dx)

		#Finding the cutoffs
		ind1 = list(map(lambda i: i < popt[0], f1)).index(True) #inclusive intensity threshold index

		try:
			ind2 = list(map(lambda i: i < der_mid, f1)).index(True) #mid intensity threshold index

		except Exception as ex:
			print(ex)
			ind2 = ind1

		#Rescaling
		thres_incl = scale(i = x[ind1], reverse=True, raw=xdata_raw)
		thres_mid = scale(i = x[ind2], reverse=True, raw=xdata_raw)
		xres = scale(i = x, reverse=True, raw=xdata_raw) #for plotting purposes

		#Assign intensity ranks
		data['_'.join(['intens_rank', marker_dict[marker]])] = data['_'.join([measurement_x, marker_dict[marker]])].rank(ascending=False, method='first')

		#Selecting data for skewness thresholding
		if len(data) > 600:

			if data.loc[data['_'.join(['intens_rank', marker_dict[marker]])] == 600.0, '_'.join([measurement_x, marker_dict[marker]])].to_list()[0] > thres_mid:
				skew_data = data.loc[data['_'.join(['intens_rank', marker_dict[marker]])] <= 600, '_'.join(['skewness', marker_dict[marker]])]
				print("Using first 600 cells for skewness filtering.")
			else:
				skew_data = data.loc[data['_'.join([measurement_x, marker_dict[marker]])] > thres_mid, '_'.join(['skewness', marker_dict[marker]])]
		
		else:
			skew_data = data.loc[data['_'.join([measurement_x, marker_dict[marker]])] > thres_mid, '_'.join(['skewness', marker_dict[marker]])]

		#Skewness threshold
		try:
			skew_thres, peak_skew, _2, _3, _4, _5 = FWHM(marker=marker, data=skew_data, scale_factor=skewness_factors[marker])
		
		except Exception as ex:
			print(ex, "Skewness thresholding did not work, set manually in GUI!")
			skew_thres = 1

		#1st graphics
		if graphics_save or graphics_show:

			#Main curve-fitting plot (x vs. y measurements)
			plt.scatter(xdata_raw, ydata, c=data['_'.join(['skewness', marker_dict[marker]])], alpha=.05)
			plt.plot(xres, y, color='red', alpha=1)
			plt.plot(xres, f1, color='green', alpha=1)
			plt.axvline(x=thres_incl, color='blue')
			plt.axvline(x=thres_mid, color='blue')
			plt.title('Curve fit 1 - ' + marker)
			plt.xlabel(measurement_x + ' (raw)')
			plt.ylabel(measurement_y + ' (scaled)')
			plt.ylim((0,1))
			ax = plt.gca()
			ax.set_facecolor('xkcd:white')
			if graphics_save:
				plt.savefig(plot_dir + '/Curvefit_plot1_' + marker +'_'+ measurement_x + '.png')
			if graphics_show:
				plt.show()
			plt.close()

			#Skewness histogram wo. threshold
			plt.hist(data['_'.join(['skewness', marker_dict[marker]])], bins=50)
			plt.axvline(x=skew_thres, color='red')
			plt.title('Skewness - ' + marker)
			ax = plt.gca()
			ax.set_facecolor('xkcd:white')
			if graphics_save:
				plt.savefig(plot_dir + '/Skewness_histo_all_' + marker + '.png')
			if graphics_show:
				plt.show()
			plt.close()

			#Skewness histogram w. threshold
			plt.hist(skew_data, bins=50)
			plt.axvline(x=skew_thres, color='red')
			plt.title('Skewness 2 - ' + marker)
			ax = plt.gca()
			ax.set_facecolor('xkcd:white')
			if graphics_save:
				plt.savefig(plot_dir + '/Skewness_histo_' + marker + '.png')
			if graphics_show:
				plt.show()
			plt.close()


		#Second iteration

		#Load updated/pre-thresholded data
		data_new = data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, ]
		xdata_new = scale(data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_x, marker_dict[marker]])])
		ydata_new = scale(data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_y, marker_dict[marker]])])
		xdata_raw_new = data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_x, marker_dict[marker]])]
		ydata_raw_new = data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_y, marker_dict[marker]])]

		#2nd fitting (on scaled data)
		popt, pcov = curve_fit(f=negative_exponential,
							   jac=negative_exponential_jacobian,
							   xdata=xdata_new,
							   ydata=ydata_new,
							   method='lm', #Levenberg-Marquard Algorithm
							   p0=[0.9, 10]) #initial guess: 0.9 for asymptote, 10 for growth rate

		x = np.linspace(0, 1, len(xdata_new))
		y = negative_exponential(x, *popt)

		#1st derivative
		dx = x[1]-x[0]
		f1 = np.gradient(y, dx)

		#Finding the cutoffs
		ind1 = list(map(lambda i: i < popt[0], f1)).index(True) #inclusive intensity threshold index
		try:
			ind2 = list(map(lambda i: i < der_mid, f1)).index(True) #mid intensity threshold index
		except Exception as ex:
			print(ex)
			ind2 = ind1

		#Rescaling
		thres_incl = scale(i = x[ind1], reverse=True, raw=xdata_raw_new)
		thres_mid = scale(i = x[ind2], reverse=True, raw=xdata_raw_new)
		xres = scale(i = x, reverse=True, raw=xdata_raw_new) #for plotting purposes

		#Consider aberrant scaling factors
		if marker in aberrant_scale.keys():
			thres_incl = thres_incl * aberrant_scale[marker]

		#Assign intensity ranks
		data_new['_'.join(['intens_rank', marker_dict[marker]])] = data_new['_'.join([measurement_x, marker_dict[marker]])].rank(ascending=False, method='first')

		#Selecting data for skewness thresholding
		if len(data_new) > 600:

			if data_new.loc[data_new['_'.join(['intens_rank', marker_dict[marker]])] == 600.0, '_'.join([measurement_x, marker_dict[marker]])].to_list()[0] > thres_mid:
				prop_data = data_new.loc[(data_new['_'.join(['intens_rank', marker_dict[marker]])] <= 600), '_'.join(['prop_50', marker_dict[marker]])]
				print("Using first 600 cells for prop50 filtering.")
			else:
				prop_data = data_new.loc[data_new['_'.join([measurement_x, marker_dict[marker]])] > thres_mid, '_'.join(['prop_50', marker_dict[marker]])]
		
		else:
			prop_data = data_new.loc[data_new['_'.join([measurement_x, marker_dict[marker]])] > thres_mid, '_'.join(['prop_50', marker_dict[marker]])]

		#Prop50 threshold
		try:
			prop_thres, peak_prop, _2, _3, _4, _5 = FWHM(marker=marker, data=prop_data, asym='right', scale_factor=0.9)
		
		except Exception as ex:
			print(ex, "Prop50 thresholding did not work, set manually in GUI!")
			prop_thres = 0.5

		#Save thresholds
		thres_intens[marker] = [thres_incl, 2.5*thres_incl]
		thres_skew[marker] = skew_thres
		thres_ent[marker] = [prop_thres, scale(i = popt[0], reverse=True, raw=ydata_raw_new)]
		exp_params2[marker] = popt

		#2nd graphics
		if graphics_save or graphics_show:

			#Main curve-fitting plot (x vs. y measurements)
			plt.scatter(xdata_raw_new, ydata_new, c=data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join(['skewness', marker_dict[marker]])], alpha=.05)
			plt.plot(xres, y, color='red', alpha=1)
			plt.plot(xres, f1, color='green', alpha=1)
			plt.axvline(x=thres_incl, color='blue')
			plt.axvline(x=2.5*thres_incl, color='blue')
			plt.axvline(x=thres_mid, color='blue')
			plt.title('Curve fit 2 - ' + marker)
			plt.xlabel(measurement_x + ' (raw)')
			plt.ylabel(measurement_y + ' (scaled)')
			plt.ylim((0,1))
			ax = plt.gca()
			ax.set_facecolor('xkcd:white')
			if graphics_save:
				plt.savefig(plot_dir + '/Curvefit_plot2_' + marker +'_'+ measurement_x + '.png')
			if graphics_show:
				plt.show()
			plt.close()

			#Prop_50 histogram w. threshold
			plt.hist(prop_data, bins=50)
			plt.axvline(x=prop_thres, color='red')
			plt.title('Prop50 - ' + marker)
			ax = plt.gca()
			ax.set_facecolor('xkcd:white')
			if graphics_save:
				plt.savefig(plot_dir + '/Prop_50_histo_' + marker + '.png')
			if graphics_show:
				plt.show()
			plt.close()

			#Prop_50 histogram wo. threshold
			plt.hist(data['_'.join(['prop_50', marker_dict[marker]])], bins=50)
			plt.axvline(x=prop_thres, color='red')
			plt.title('Prop50_all - ' + marker)
			ax = plt.gca()
			ax.set_facecolor('xkcd:white')
			if graphics_save:
				plt.savefig(plot_dir + '/Prop_50_histo_all_' + marker + '.png')
			if graphics_show:
				plt.show()
			plt.close()
			
	return thres_intens, thres_skew, thres_ent, exp_params1, exp_params2 


def FWHM_thresholding(data, markers_FWHM, base_scale_factor=1, base_symmetry='sym'): #base_symmetry can be sym, left or right
	
	"""
	Applies Full Width at Half Maximum (FWHM) thresholding to a list of specified markers.
	Uses the data in log-transformed form for thresholding.

	Parameters
	----------
	data : DataFrame
		The dataset containing the measurements.
	markers_FWHM : list of str
		The list of marker names to apply FWHM thresholding.
	base_scale_factor : float, optional
		The base scale factor to adjust the thresholding calculation. Defaults to 1.
	base_symmetry : {'sym', 'left', 'right'}, optional
		The assumption about the symmetry of the data distribution. (see FWHM() function)

	Returns
	-------
	thres_FWHM : dict
		A dictionary with markers as keys and their corresponding logFWHM thresholds as values.

	Notes
	-----
	- This function calculates the FWHM threshold for each specified marker using the logFWHM function.
	- In case the FWHM calculation fails for any marker, the function sets the threshold for that marker to 1
	  and prints an error message advising downstream manual threshold setting in the GUI.
	"""
	
	thres_FWHM = {}
	
	for marker in markers_FWHM:

		try:
			thres_marker, _1,_2,_3,_4,_5 = logFWHM(marker, data['_'.join([intensity_measure, marker_dict[marker]])], asym=base_symmetry, scale_factor=base_scale_factor)

		except Exception as ex:
			thres_FWHM[marker] = 1
			print(ex, "FWHM thresholding didn't work! Set manually in GUI.")

		else:
			thres_FWHM[marker] = thres_marker
 
	return thres_FWHM


def autofluo_thresholding(data, columns_autofluo, base_scale_factor=1, base_symmetry='sym'): #base_symmetry can be sym, left or right
	
	'''Very similar to FWHM_thresholding().
	   Specifically meant for thresholding autofluorescence levels for QC purposes.
	   Only difference: Uses raw (non-log) data for thresholding with FWHM().'''
	
	thres_FWHM = {}
	
	for marker in columns_autofluo:
		
		try:
			thres_marker, _1,_2,_3,_4,_5 = FWHM(marker, data[marker], asym=base_symmetry, scale_factor=base_scale_factor)
		
		except Exception as ex:
			thres_FWHM[marker] = 1
			print(ex, "FWHM thresholding didn't work! Set manually.")
		
		else:
			thres_FWHM[marker] = thres_marker
 
	return thres_FWHM


def tempo_idents_exp(data, thres_intens, thres_skew, thres_ent, exp_params, prop_thres_strictness): 

	"""
	Creates a list of binary cellular identities (pos/neg) for a given marker that is to be thresholded by the combined / negative exponential fit method.
	Used for displaying purposes in manual GUI thresholding.

	Parameters
	----------
	data : DataFrame
		Pandas dataframe containing all single cell level data to be processed.
	thres_intens : tuple
		A tuple of (lower, upper) intensity thresholds.
	thres_skew : float
		The skewness threshold.
	thres_ent : tuple
		A tuple of (base, upper) entropy thresholds.
	exp_params : tuple
		Parameters for the negative exponential scaling function (used for data scaling).
	prop_thres_strictness : {'high', 'mid', 'low'}
		The stringency of prop_50 thresholding: affects how prop_50 thresholds are calculated.

	Returns
	-------
	raw_ident_list : list of str
		A list of raw identity strings ('pos' or 'neg') for each cell.
	"""
	
	raw_ident_list = []
	scale_data_dict = dict(zip(data['unique_ident'], scale(data['intensity'])))
	  
	#Iterate through all cells
	for cell in data['unique_ident']:
			
		#Collect data
		intens = data.loc[data['unique_ident'] == cell, 'intensity'].to_list()[0]
		prop = data.loc[data['unique_ident'] == cell, 'prop50'].to_list()[0]
		skew = data.loc[data['unique_ident'] == cell, 'skewness'].to_list()[0]
			
		#Prepare dynamic prop50 threshold depending on selected stringency
		prop_thres_width = thres_intens[1] - thres_intens[0]
			
		if (intens - thres_intens[1]) < 0:
			prop_thres_diff = thres_ent[0] - thres_ent[1]
			if prop_thres_strictness == 'high':
				prop50 = (((intens - thres_intens[0]) / prop_thres_width) * 0.5 * prop_thres_diff) + scale(i = negative_exponential(scale_data_dict[cell], *exp_params), reverse=True, raw=data['prop50']) + 0.5 * prop_thres_diff
			elif prop_thres_strictness == 'mid':
				prop50 = prop_thres_diff + scale(i = negative_exponential(scale_data_dict[cell], *exp_params), reverse=True, raw=data['prop50'])
			elif prop_thres_strictness =='low':
				prop50 = thres_ent[0]
		else:
			prop50 = 1
		
		#Apply thresholds and determine marker positivity
		if (prop < prop50) & (intens > thres_intens[0]) & (skew < thres_skew):
			 raw_ident_list.append('pos')
		else:
			raw_ident_list.append('neg')
	
	return raw_ident_list


def tempo_idents_FWHM(data, thres_FWHM): 

	"""
	Creates a list of binary cellular identities (pos/neg) for a given marker that is to be thresholded by the Full Width at Half Maximum (FWHM) method.
	Used for displaying purposes in manual GUI thresholding.

	Parameters
	----------
	data : DataFrame
		Pandas dataframe containing all single cell level data to be processed.
	thres_FWHM : tuple
		The intensity threshold.

	Returns
	-------
	raw_ident_list : list of str
		A list of raw identity strings ('pos' or 'neg') for each cell.
	"""
	
	raw_ident_list = []      
		
	#Iterate through all cells
	for cell in data['unique_ident']:

		#Collect data
		intens = data.loc[data['unique_ident'] == cell, 'intensity'].to_list()[0]

		#Apply thresholds and determine marker positivity
		if intens > thres_FWHM:
			raw_ident_list.append('pos')
		else:
			raw_ident_list.append('neg')

	return raw_ident_list


def raw_idents(data, marker_dict, thres_intens, thres_skew, thres_ent, exp_params, thres_FWHM, prop_thres_strictness): 

	"""
	Determines raw cellular identities (e.g. positive markers) based on previously defined thresholds.

	Parameters
	----------
	data : DataFrame
		Pandas dataframe containing all single cell level data to be processed.
	marker_dict : dict
		A dictionary mapping marker names to their respective column identifiers in `data` (e.g. cycle and channel).
	thres_intens : dict
		Intensity thresholds for each marker.
	thres_skew : dict
		Skewness thresholds for each marker.
	thres_ent : dict
		Prop_50/entropy thresholds for each marker.
	exp_params : dict
		Parameters for the negative exponential function (used for data scaling).
	thres_FWHM : dict
		Full Width at Half Maximum thresholds for each marker.
	prop_thres_strictness : {'high', 'mid', 'low'}
		The stringency of prop_50 thresholding: affects how prop_50 thresholds are calculated.

	Returns
	-------
	raw_ident_list : list of str
		A list of raw identity strings for each cell in `data`. Used downstream for phenotype assignment.
	"""
	
	raw_ident_list = []
	ident_list = []
	scale_data_dict = {}
	
	#Create scaled versions of the intensity data
	for m in marker_dict.keys():
		scale_data_dict[m] = dict(zip(data['unique_ident'], scale(data['_'.join([intensity_measure, marker_dict[m]])])))
	  
	#Process all cells individually
	for cell in data['unique_ident']:
		
		temp_pos_markers = []

		#Handle negative exponential thresholded markers
		for marker, thresholds in thres_intens.items():
			
			#Collect data
			intens = data.loc[data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict[marker]])].to_list()[0]
			prop = data.loc[data['unique_ident'] == cell, '_'.join(['prop_50', marker_dict[marker]])].to_list()[0]
			skew = data.loc[data['unique_ident'] == cell, '_'.join(['skewness', marker_dict[marker]])].to_list()[0]
			
			#Prepare dynamic prop50 threshold depending on selected stringency
			prop_thres_width = thresholds[1] - thresholds[0]
			
			if (intens - thresholds[1]) < 0:
				prop_thres_diff = thres_ent[marker][0] - thres_ent[marker][1]
				
				if prop_thres_strictness == 'high':
					prop50 = ((intens - thresholds[0]) / prop_thres_width) * 0.5 * prop_thres_diff + scale(i = negative_exponential(scale_data_dict[marker][cell], *exp_params[marker]), reverse=True, raw=data['_'.join(['prop_50', marker_dict[marker]])]) + 0.5 * prop_thres_diff
				
				elif prop_thres_strictness == 'mid':
					prop50 = prop_thres_diff + scale(i = negative_exponential(scale_data_dict[marker][cell], *exp_params[marker]), reverse=True, raw=data['_'.join(['prop_50', marker_dict[marker]])])
				
				elif prop_thres_strictness =='low':
					prop50 = thres_ent[marker][0]

			else:
				prop50 = 1


			#Apply thresholds and determine marker positivity
			if (prop < prop50) & (intens > thresholds[0]) & (skew < thres_skew[marker]):
				temp_pos_markers.append(marker)

		#Handle FWHM thresholded markers
		for marker, thresholds in thres_FWHM.items():
			intens = data.loc[data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict[marker]])].to_list()[0]
			
			#Apply threshold and determine marker positivity
			if intens > thresholds:
				temp_pos_markers.append(marker)

		
		#Cells that are not positive for any marker --> undefined
		if len(temp_pos_markers) == 0:
			raw_ident_list.append('undefined')

		#Cells that are positive for at least one marker   
		elif len(temp_pos_markers) == 1:
			raw_ident_list.append(temp_pos_markers[0])
		else:
			raw_ident_list.append('_'.join(temp_pos_markers))

	
	return raw_ident_list


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


##HELPER FUNCTIONS (to be used with map())

def posneg(val):

	if val == 'pos':
		return 1
	else:
		return 0
	

def KI67(Mstring):

	if 'KI67' in Mstring:
		return 'pos'
	else:
		return 'neg'



##DATA STRUCTURES FOR GUI

class GUI_FWHM(Tk):
	
	"""
	A GUI class for manual thresholding QC of FWHM (Full Width at Half Maximum) thresholded markers.
	
	Attributes:
		in_data (DataFrame): Main input dataframe containing intensity and other measurements.
		thres_intens (float): Initial intensity threshold value.
		FOV_ranks (list): Ordered list of Field of Views (FOV) based on certain criteria (e.g. cell count).
		FOV_dict (dict): Dictionary mapping FOVs to their image file paths.
		seg_dict (dict): Dictionary mapping FOVs to their segmentation mask file paths.
		marker (str): The marker name for which the thresholding is being performed.
	"""
	
	def __init__(self, in_data, thres_intens, FOV_ranks, FOV_dict, seg_dict, marker, *args, **kwargs):

		"""Initializes the GUI window, loads input data, and sets up the UI components."""
		
		#INHERITANCE
		super().__init__(*args, **kwargs)
		
		#BASIC WINDOW GEOMETRY
		self.title('Manual thresholding - '+marker)
		self.geometry('1600x800')
		self.protocol('WM_DELETE_WINDOW', self.window_del)
		
		#INTRA-WINDOW GEOMETRY
		for i in range(0,3):
			self.columnconfigure(i, weight=1, minsize=300)
		for i in range(0,5):
			self.rowconfigure(i, weight=1)

		self.dropframe1 = Frame(self)
		self.dropframe2 = Frame(self)
		self.dropframe3 = Frame(self)
		self.dropframe1.grid(row=0, columnspan=1, column=0)
		self.dropframe2.grid(row=0, columnspan=1, column=1)
		self.dropframe3.grid(row=0, columnspan=1, column=2)

		self.imageframe1 = Frame(self)
		self.imageframe2 = Frame(self)
		self.imageframe3 = Frame(self)
		self.imageframe1.grid(row=1, columnspan=1, column=0)
		self.imageframe2.grid(row=1, columnspan=1, column=1)
		self.imageframe3.grid(row=1, columnspan=1, column=2)

		self.setframe12 = Frame(self)
		self.setframe22 = Frame(self)
		self.setframe12.grid(row=2, columnspan=1, column=1)
		self.setframe22.grid(row=3, columnspan=1, column=1)

		self.saveframe = Frame(self)
		self.saveframe.grid(row=4, columnspan=3, column=0)
		
		#LOAD INPUTS
		self.in_data = in_data
		self.thres_intens = thres_intens
		self.FOV_ranks = FOV_ranks
		self.FOV_dict = FOV_dict
		self.seg_dict = seg_dict
		
		#INITIALIZE DYNAMIC VARIABLES
		self.intVar = DoubleVar()
		self.intVar.set(self.thres_intens)
		
		self.brightVar = DoubleVar()
		self.brightVar.set(2)
		
		self.currFOV = StringVar()
		self.currFOV.set(self.FOV_ranks[0])
		
		self.intensity = np.array(self.in_data['intensity'])
		self.intensity = self.intensity[~np.isnan(self.intensity)]
		
		#INITIALIZE TRACKING VARIABLES
		self.track_int = self.intVar.get()
		self.track_bright =  self.brightVar.get()
		self.track_currFOV = self.currFOV.get()
		
		#SETUP WIDGETS
	
		#TOP ROW
		#Refresh button
		self.refreshbutton = ttk.Button(master=self.saveframe, text='Refresh', style='save.TButton', command=self.update_window)
		self.refreshbutton.pack()
		
		#Reset button
		self.resetbutton = ttk.Button(master=self.saveframe, text='Reset', style='save.TButton', command=self.reset_thres)
		self.resetbutton.pack()
		
		self.buttonStyle = ttk.Style()
		self.buttonStyle.configure('save.TButton', width=80)

		#BOTTOM ROW
		#FOV selection static Text Label
		self.FOV_text = ttk.Label(self.dropframe3, text='FOV selector')
		self.FOV_text.pack()
		
		#FOV drop-down menu
		self.status = ttk.Label(self.dropframe3, text='Ready', foreground='#009933')
		self.status.pack(side=RIGHT, padx=15)
		self.FOV_list = ttk.Combobox(self.dropframe3, textvariable = self.currFOV)
		self.FOV_list['values'] = list(self.FOV_dict.keys())
		self.FOV_list.pack(side=RIGHT, padx=15)
		
		#Display highest and lowest cell count FOVs (static)
		self.FOV_show = ttk.Label(self.dropframe1, text=f'High count FOVs: {self.FOV_ranks[0:7]}\nLow count FOVs: {self.FOV_ranks[-7:-1]}')
		self.FOV_show.pack()
		
		#CHANGING THRESHOLDS
		self.labInt = ttk.Label(self.setframe12, text='Intensity')
		self.labInt.pack()
		self.scaleInt = ttk.Scale(self.setframe12, orient=HORIZONTAL, length=350, from_=min(self.intensity), to=max(self.intensity), variable=self.intVar)
		self.scaleInt.pack()
		self.spinInt = ttk.Spinbox(self.setframe12, from_=min(self.intensity), to=max(self.intensity), textvariable=self.intVar, increment=1)
		self.spinInt.pack()

		#CHANGING IMAGE BRIGHTNESS
		self.labBright = ttk.Label(self.dropframe2, text='Brightness')
		self.labBright.pack()
		self.scaleBright = ttk.Spinbox(self.dropframe2, from_=0, to=30, increment=0.1, textvariable=self.brightVar)
		self.scaleBright.pack()
		
		#IMAGES/GRAPHS
		#Initialize column 'preID'
		self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
		self.in_data_sel['preID'] = tempo_idents_FWHM(self.in_data_sel, thres_FWHM = self.intVar.get())

		#Initialze qsub and seg images
		self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
		self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')

		#IMAGES
		self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
		self.im_arr = self.im_arr.astype(np.uint8)
		self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])
		
		self.im2 = Img.fromarray(self.im_arr)
		self.im2en = ImageEnhance.Brightness(self.im2)
		self.im2 = self.im2en.enhance(self.brightVar.get())  

		self.python_im2 = ImageTk.PhotoImage(self.im2, master=self.imageframe2)
		self.mid_im = Label(self.imageframe2,  image=self.python_im2)
		self.mid_im.pack()
		
		#Initialize numeric IDs
		self.in_data_sel['num_ident'] = np.ones((len(self.in_data_sel)), dtype='int32')
		self.in_data_sel.loc[self.in_data_sel['preID'] == 'pos', 'num_ident'] = 2

		self.circle_mask = annotate(self.start_seg, dotsize=2)
		self.one_hot_hm = mask_heatmap(self.circle_mask, self.in_data_sel, 'num_ident')
		self.im_arr2 = self.im_arr.copy()
		self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
		self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
		self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
		self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

		self.Hm_rgb = Img.fromarray(self.im_arr2)

		self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb, master=self.imageframe3)
		self.right_im = Label(self.imageframe3,  image=self.python_im3)
		self.right_im.pack()

		#INTERACTIVE HISTOGRAM
		self.fig = Figure(figsize = (4, 4), dpi = 100)
		self.plot = self.fig.add_subplot(111)
		self.plot.hist(self.in_data_sel['intensity'], bins=50)
		self.plot.axvline(x=self.intVar.get(), c='red')

		self.canvas = FigureCanvasTkAgg(self.fig, self.imageframe1)  
		self.canvas.draw()
		self.canvas.get_tk_widget().pack()

		self.toolbar = NavigationToolbar2Tk(self.canvas, self.imageframe1)
		self.toolbar.update()
		self.canvas.get_tk_widget().pack()
		
		#INITIALIZATION DONE
		
		
	def window_del(self):

		"""Saves the current intensity threshold to a newly created global variable and closes the window."""

		global intens_thres 
		intens_thres = self.intVar.get()
		
		self.destroy()
					 

	def reset_thres(self):

		"""Resets the intensity threshold to the initial value provided at startup."""

		self.intVar.set(self.thres_intens)

		
	def update_window(self):
		
		"""Refreshes the window to apply any changes made to the threshold, brightness or selected FOV."""

		self.reload_win()

		
	def reload_win(self):

		"""Recomputes and updates the displayed data based on current settings (e.g., selected FOV, intensity threshold, brightness).
		This includes updating the images, the heatmap, and the histogram plot to reflect the current thresholding and FOV selection."""
	
		#Set status to 'Wait'
		self.status.config(text='Wait', foreground='#ff0000')
		
		self.IDnew = False
		self.FOVnew = False
		
		#Handle FOV changes
		if self.track_currFOV != self.currFOV.get():
			
			self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
			
			self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
			self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')
			
			self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
			self.im_arr = self.im_arr.astype(np.uint8)
			self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])

			self.im2 = Img.fromarray(self.im_arr)
			self.im2en = ImageEnhance.Brightness(self.im2)
			self.im2 = self.im2en.enhance(self.brightVar.get())

			self.python_im2 = ImageTk.PhotoImage(self.im2)
			self.mid_im.config(image=self.python_im2)
			self.mid_im.image = self.python_im2
			self.FOVnew = True
			
		#Handle brightness changes
		elif self.FOVnew == False and self.track_bright != self.brightVar.get():
			
			self.im2 = Img.fromarray(self.im_arr)
			self.im2en = ImageEnhance.Brightness(self.im2)
			self.im2 = self.im2en.enhance(self.brightVar.get())

			self.python_im2 = ImageTk.PhotoImage(self.im2)
			self.mid_im.config(image=self.python_im2)
			self.mid_im.image = self.python_im2
			self.brightnew = True
			
		#Re-process 'preID' column in case of FOV or threshold change
		if self.track_int != self.intVar.get() or self.track_currFOV != self.currFOV.get():
			
			self.in_data_sel['preID'] = tempo_idents_FWHM(self.in_data_sel, thres_FWHM = self.intVar.get())
			self.IDnew = True

		#Update histogram and synthetic image in case of FOV or threshold change
		if self.FOVnew == True or self.IDnew == True:
			
			self.in_data_sel['num_ident'] = np.ones((len(self.in_data_sel)), dtype='int32')
			self.in_data_sel.loc[self.in_data_sel['preID'] == 'pos', 'num_ident'] = 2

			self.circle_mask = annotate(self.start_seg, dotsize=2)
			self.one_hot_hm = mask_heatmap(self.circle_mask, self.in_data_sel, 'num_ident')
			
			self.im_arr2 = self.im_arr.copy()
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

			self.Hm_rgb = Img.fromarray(self.im_arr2)
			self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
			self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

			self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
			self.right_im.config(image=self.python_im3)
			self.right_im.image = self.python_im3

			#Update plot
			self.fig.clear()
			self.plot = self.fig.add_subplot(111)
			self.plot.hist(self.in_data_sel['intensity'], bins=50)
			self.plot.axvline(x=self.intVar.get(), c='red')

			self.canvas.draw_idle()
			self.toolbar.update()
			
		#Update synthetic image in case of brightness change
		elif self.FOVnew == False and self.IDnew == False and self.brightnew == True:
			
			self.im_arr2 = self.im_arr.copy()
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

			self.Hm_rgb = Img.fromarray(self.im_arr2)
			self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
			self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

			self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
			self.right_im.config(image=self.python_im3)
			self.right_im.image = self.python_im3
			
		#Set status to 'Ready'
		self.status.config(text='Ready', foreground='#009933')
		
		#Update tracking variables
		self.track_int = self.intVar.get()
		self.track_currFOV = self.currFOV.get()
		self.track_bright = self.brightVar.get()
		
	
class GUI_exp(Tk): 
	
	"""
	A GUI class for manual thresholding QC of negative exponential fit thresholded markers (combined thresholding).
	
	Attributes:
		in_data (DataFrame): Main input dataframe containing intensity and other measurements.
		thres_intens (list, len=2): Initial intensity threshold value list.
		thres_skew (float): Inital skewness threshold value.
		thres_ent (list, len=2): Initial entropy/prop_50 threshold value list.
		exp_params (list): Parameters from second negative exponential fit in combined_thresholding().
		prop_thres_strictness ({'high', 'mid', 'low'}): Stringency of prop_50 thresholding.
		FOV_ranks (list): Ordered list of Field of Views (FOV) based on certain criteria (e.g. cell count).
		FOV_dict (dict): Dictionary mapping FOVs to their image file paths.
		seg_dict (dict): Dictionary mapping FOVs to their segmentation mask file paths.
		marker (str): The marker name for which the thresholding is being performed.
	"""
	
	def __init__(self, in_data, thres_intens, thres_skew, thres_ent, exp_params, prop_thres_strictness,
				 FOV_ranks, FOV_dict, seg_dict, marker, *args, **kwargs):
		
		#INHERITANCE
		super().__init__(*args, **kwargs)
		
		#BASIC WINDOW GEOMETRY
		self.title('Manual thresholding - '+marker)
		self.geometry('1600x800')
		self.protocol('WM_DELETE_WINDOW', self.window_del)
		
		#INTRA-WINDOW GEOMETRY
		for i in range(0,3):
			self.columnconfigure(i, weight=1, minsize=300)
		for i in range(0,5):
			self.rowconfigure(i, weight=1)

		self.dropframe1 = Frame(self)
		self.dropframe2 = Frame(self)
		self.dropframe3 = Frame(self)
		self.dropframe1.grid(row=0, columnspan=1, column=0)
		self.dropframe2.grid(row=0, columnspan=1, column=1)
		self.dropframe3.grid(row=0, columnspan=1, column=2)

		self.imageframe1 = Frame(self)
		self.imageframe2 = Frame(self)
		self.imageframe3 = Frame(self)
		self.imageframe1.grid(row=1, columnspan=1, column=0)
		self.imageframe2.grid(row=1, columnspan=1, column=1)
		self.imageframe3.grid(row=1, columnspan=1, column=2)

		self.setframe11 = Frame(self)
		self.setframe12 = Frame(self)
		self.setframe13 = Frame(self)
		self.setframe21 = Frame(self)
		self.setframe22 = Frame(self)
		self.setframe23 = Frame(self)
		self.setframe11.grid(row=2, columnspan=1, column=0)
		self.setframe12.grid(row=2, columnspan=1, column=1)
		self.setframe13.grid(row=2, columnspan=1, column=2)
		self.setframe21.grid(row=3, columnspan=1, column=0)
		self.setframe22.grid(row=3, columnspan=1, column=1)
		self.setframe23.grid(row=3, columnspan=1, column=2)

		self.saveframe = Frame(self)
		self.saveframe.grid(row=4, columnspan=3, column=0)
		
		#LOAD INPUTS
		self.in_data = in_data
		self.thres_intens = thres_intens
		self.thres_skew = thres_skew
		self.thres_prop = thres_ent
		self.exp_params = exp_params
		self.prop_thres_strictness = prop_thres_strictness
		self.FOV_ranks = FOV_ranks
		self.FOV_dict = FOV_dict
		self.seg_dict = seg_dict
		
		self.intensity = np.array(self.in_data['intensity'])
		self.intensity = self.intensity[~np.isnan(self.intensity)]
		
		self.skewness = np.array(self.in_data['skewness'])
		self.skewness = self.skewness[~np.isnan(self.skewness)]
		
		#INITIALIZE DYNAMIC VARIABLES
		self.intVar = DoubleVar()
		self.intVar.set(self.thres_intens[0])
		
		self.skewVar = DoubleVar()
		self.skewVar.set(self.thres_skew)
		
		self.propVar = DoubleVar()
		self.propVar.set(self.thres_prop[0] - self.thres_prop[1])
		
		self.brightVar = DoubleVar()
		self.brightVar.set(2)
		
		self.currFOV = StringVar()
		self.currFOV.set(self.FOV_ranks[0])
	
		
		#INITIALIZE TRACKING VARIABLES
		self.track_int = self.intVar.get()
		self.track_skew = self.skewVar.get()
		self.track_prop = self.propVar.get()
		self.track_bright =  self.brightVar.get()
		self.track_currFOV = self.currFOV.get()
		
		#WIDGETS
	
		#TOP ROW
		#Refresh button
		self.refreshbutton = ttk.Button(master=self.saveframe, text='Refresh', style='save.TButton', command=self.update_window)
		self.refreshbutton.pack()
		
		#Reset button
		self.resetbutton = ttk.Button(master=self.saveframe, text='Reset', style='save.TButton', command=self.reset_thres)
		self.resetbutton.pack()

		self.buttonStyle = ttk.Style()
		self.buttonStyle.configure('save.TButton', width=80)

		#BOTTOM ROW
		#FOV selection static Text Label
		self.FOV_text = ttk.Label(self.dropframe3, text='FOV selector')
		self.FOV_text.pack()

		#FOV drop-down menu
		self.status = ttk.Label(self.dropframe3, text='Ready', foreground='#009933')
		self.status.pack(side=RIGHT, padx=15)
		self.FOV_list = ttk.Combobox(self.dropframe3, textvariable = self.currFOV)
		self.FOV_list['values'] = list(self.FOV_dict.keys())
		self.FOV_list.pack(side=RIGHT, padx=15)
		
		#Display high and low cell count FOVs (static)
		self.FOV_show = ttk.Label(self.dropframe1, text=f'High count FOVs: {self.FOV_ranks[0:7]}\nLow count FOVs: {self.FOV_ranks[-7:-1]}')
		self.FOV_show.pack()
		
		#CHANGING THRESHOLDS
		self.labInt = ttk.Label(self.setframe11, text='Intensity')
		self.labInt.pack()
		self.scaleInt = ttk.Scale(self.setframe11, orient=HORIZONTAL, length=350, from_=min(self.intensity), to=max(self.intensity), variable=self.intVar)
		self.scaleInt.pack()
		self.spinInt = ttk.Spinbox(self.setframe11, from_=min(self.intensity), to=max(self.intensity), textvariable=self.intVar, increment=1)
		self.spinInt.pack()

		self.labProp = ttk.Label(self.setframe12, text='Prop_50')
		self.labProp.pack()
		self.scaleProp = ttk.Scale(self.setframe12, orient=HORIZONTAL, length=350, from_=0, to=self.propVar.get()*3, variable=self.propVar)
		self.scaleProp.pack()
		self.spinProp = ttk.Spinbox(self.setframe12, from_=0, to=self.propVar.get()*3, textvariable=self.propVar, increment=0.01)
		self.spinProp.pack()

		self.labSkew = ttk.Label(self.setframe13, text='Skewness')
		self.labSkew.pack()
		self.scaleSkew = ttk.Scale(self.setframe13, orient=HORIZONTAL, length=350, from_=min(self.skewness), to=max(self.skewness), variable=self.skewVar)
		self.scaleSkew.pack()
		self.spinSkew = ttk.Spinbox(self.setframe13, from_=min(self.skewness), to=max(self.skewness), textvariable=self.skewVar, increment=0.01)
		self.spinSkew.pack()
		
		#CHANGING IMAGE BRIGHTNESS
		self.labBright = ttk.Label(self.dropframe2, text='Brightness')
		self.labBright.pack()
		self.scaleBright = ttk.Spinbox(self.dropframe2, from_=0, to=30, increment=0.1, textvariable=self.brightVar)
		self.scaleBright.pack()
		
		#IMAGES/GRAPHS
		#Initialize column 'preID'
		self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
		self.in_data_sel['preID'] = tempo_idents_exp(self.in_data_sel, 
											thres_intens = [self.intVar.get(), self.intVar.get()*2.5], 
											thres_skew = self.skewVar.get(), 
											thres_ent = [self.thres_prop[1]+self.propVar.get(), self.thres_prop[1]], 
											exp_params = self.exp_params, 
											prop_thres_strictness = self.prop_thres_strictness)

		#Initialize first qsub and seg images
		self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
		self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')

		#IMAGES
		self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
		self.im_arr = self.im_arr.astype(np.uint8)
		self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])
		
		self.im2 = Img.fromarray(self.im_arr)
		self.im2en = ImageEnhance.Brightness(self.im2)
		self.im2 = self.im2en.enhance(self.brightVar.get())

		self.python_im2 = ImageTk.PhotoImage(self.im2, master=self.imageframe2)
		self.mid_im = Label(self.imageframe2,  image=self.python_im2)
		self.mid_im.pack()
		
		#Initialize numeric IDs
		self.in_data_sel['num_ident'] = np.ones((len(self.in_data_sel)), dtype='int32') #initialize numeric IDs
		self.in_data_sel.loc[self.in_data_sel['preID'] == 'pos', 'num_ident'] = 2

		self.circle_mask = annotate(self.start_seg, dotsize=2)
		self.one_hot_hm = mask_heatmap(self.circle_mask, self.in_data_sel, 'num_ident')
		self.im_arr2 = self.im_arr.copy()
		self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
		self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
		self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
		self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

		self.Hm_rgb = Img.fromarray(self.im_arr2)

		self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb, master=self.imageframe3)
		self.right_im = Label(self.imageframe3,  image=self.python_im3)
		self.right_im.pack()

		#INTERACTIVE SCATTERPLOT
		self.fig = Figure(figsize = (4, 4), dpi = 100)
		self.plot = self.fig.add_subplot(111)
		self.plot.scatter(x = self.in_data_sel['intensity'], 
						  y = self.in_data_sel['prop50'],
						  c = list(map(posneg, self.in_data_sel['preID'])),
						  alpha = 0.5)

		self.canvas = FigureCanvasTkAgg(self.fig, self.imageframe1)  
		self.canvas.draw()
		self.canvas.get_tk_widget().pack()

		self.toolbar = NavigationToolbar2Tk(self.canvas, self.imageframe1)
		self.toolbar.update()
		self.canvas.get_tk_widget().pack()
		
		#INITIALIZATION DONE
		
		
	def window_del(self):

		"""Saves the current intensity threshold to a newly created global variable and closes the window."""

		global intens_thres 
		intens_thres = [self.intVar.get(), self.intVar.get()*2.5]
		
		global skew_thres 
		skew_thres = self.skewVar.get()
		
		global ent_thres 
		ent_thres = [self.thres_prop[1]+self.propVar.get(), self.thres_prop[1]]
		
		self.destroy()
   
					 
	def reset_thres(self):

		"""Resets the intensity threshold to the initial value provided at startup."""

		self.intVar.set(self.thres_intens[0])
		self.skewVar.set(self.thres_skew)
		self.propVar.set(self.thres_prop[0] - self.thres_prop[1])
		

	def update_window(self):
		
		"""Refreshes the window to apply any changes made to the threshold, brightness or selected FOV."""

		self.reload_win()
		

	def reload_win(self):

		"""Recomputes and updates the displayed data based on current settings (e.g., selected FOV, thresholds, brightness).
		This includes updating the images, the heatmap, and the histogram plot to reflect the current thresholding and FOV selection."""
	
		#Set status to 'Wait'
		self.status.config(text='Wait', foreground='#ff0000')
		
		self.IDnew = False
		self.FOVnew = False
		
		#Handle FOV changes
		if self.track_currFOV != self.currFOV.get():
			
			self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
			
			self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
			self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')
			
			self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
			self.im_arr = self.im_arr.astype(np.uint8)
			self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])

			self.im2 = Img.fromarray(self.im_arr)
			self.im2en = ImageEnhance.Brightness(self.im2)
			self.im2 = self.im2en.enhance(self.brightVar.get())

			self.python_im2 = ImageTk.PhotoImage(self.im2)
			self.mid_im.config(image=self.python_im2)
			self.mid_im.image = self.python_im2
			self.FOVnew = True
			
		#Handle brightness changes
		elif self.FOVnew == False and self.track_bright != self.brightVar.get():
  
			self.im2 = Img.fromarray(self.im_arr)
			self.im2en = ImageEnhance.Brightness(self.im2)
			self.im2 = self.im2en.enhance(self.brightVar.get())

			self.python_im2 = ImageTk.PhotoImage(self.im2)
			self.mid_im.config(image=self.python_im2)
			self.mid_im.image = self.python_im2
			self.brightnew = True
		
		#Re-process 'preID' column in case of FOV or threshold change
		if self.track_int != self.intVar.get() or self.track_skew != self.skewVar.get() or self.track_prop != self.propVar.get() or self.track_currFOV != self.currFOV.get():
			
			self.in_data_sel['preID'] = tempo_idents_exp(self.in_data_sel, 
													 thres_intens = [self.intVar.get(), self.intVar.get()*2.5], 
													 thres_skew = self.skewVar.get(), 
													 thres_ent = [self.thres_prop[1]+self.propVar.get(), self.thres_prop[1]], 
													 exp_params = self.exp_params, 
													 prop_thres_strictness = self.prop_thres_strictness)
			
			self.IDnew = True
		
		#Update histogram and synthetic image in case of FOV or threshold change    
		if self.FOVnew == True or self.IDnew == True:
			
			self.in_data_sel['num_ident'] = np.ones((len(self.in_data_sel)), dtype='int32')
			self.in_data_sel.loc[self.in_data_sel['preID'] == 'pos', 'num_ident'] = 2

			self.circle_mask = annotate(self.start_seg, dotsize=2)
			self.one_hot_hm = mask_heatmap(self.circle_mask, self.in_data_sel, 'num_ident')
			
			self.im_arr2 = self.im_arr.copy()
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

			self.Hm_rgb = Img.fromarray(self.im_arr2)
			self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
			self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

			self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
			self.right_im.config(image=self.python_im3)
			self.right_im.image = self.python_im3

			#Update plot
			self.fig.clear()
			self.plot = self.fig.add_subplot(111)
			self.plot.scatter(x = self.in_data_sel['intensity'], 
							  y = self.in_data_sel['prop50'],
							  c = list(map(posneg, self.in_data_sel['preID'])),
							  alpha = 0.5)

			self.canvas.draw_idle()
			self.toolbar.update()
			
		#Update synthetic image in case of brightness change    
		elif self.FOVnew == False and self.IDnew == False and self.brightnew == True:
			
			self.im_arr2 = self.im_arr.copy()
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
			self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
			self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

			self.Hm_rgb = Img.fromarray(self.im_arr2)
			self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
			self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

			self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
			self.right_im.config(image=self.python_im3)
			self.right_im.image = self.python_im3
		
		#Set status to 'Ready'    
		self.status.config(text='Ready', foreground='#009933')
		
		#Update tracking vars
		self.track_int = self.intVar.get()
		self.track_skew = self.skewVar.get()
		self.track_prop = self.propVar.get()  
		self.track_currFOV = self.currFOV.get()
		self.track_bright = self.brightVar.get()