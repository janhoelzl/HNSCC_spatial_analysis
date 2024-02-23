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
    '''
    mask :: array, assumes mask background is 0
    dotsize :: radius of circles drawn in pixels
    '''
    int_mask = mask.copy()
    mask_unique = np.unique(int_mask.flatten())[1:]
    
    for c in mask_unique:
        obj_coord = np.where(int_mask==c)
        if obj_coord[0].size != 0 and obj_coord[1].size != 0:
            obj_mask = int_mask[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1]
            y_center, x_center = np.argwhere(obj_mask==c).sum(0)/(obj_mask==c).sum()
            new_entry = np.zeros(obj_mask.shape)
            rr, cc = draw.disk((y_center, x_center), radius=dotsize, shape=new_entry.shape)
            new_entry[rr, cc] = c

            int_mask[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1] = new_entry

    return int_mask


def FWHM(marker, data, asym='sym', scale_factor=1): 
    
    '''Finds peak + FWHM threshold in a given signal;
    peak is expected to be highest and leftmost -> if not -> warning message and highest peak is used.
    In case data is left-skewed, logFWHM down below will be the better option.
    Asym argument can either be None, left, or right depending on assumptions about data distribution.'''
    
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), np.array(data).size)
    y = kde.evaluate(x)
    
    peaks, properties = find_peaks(y, prominence=0)
    print(f'Peaks for {marker} are: {x[peaks]}.')
    print('Prominences: ', properties["prominences"])
    
    exp_peak = peaks[0]
    for peak in peaks:
        if kde(x[peak]) > 3*kde(x[exp_peak]):
            print("Warning: Leftmost and highest peak are not identical! Highest is used.")
            exp_peak = peak
    
    #Assumption: Hightest peak corresponds to negative population
    width, height, left_ips, right_ips = peak_widths(y, np.array([exp_peak]), rel_height=0.5)
    width = round(width[0]) * ((np.max(x) - np.min(x)) / x.size)
    left_ips = x[round(left_ips[0])]
    right_ips = x[round(right_ips[0])]
    print('width: ', width, 'left_ips: ', left_ips, 'right_ips: ', right_ips, 'peak: ', x[exp_peak])
    
    if asym == 'left':
        thres = x[exp_peak] + scale_factor * (2 * ((x[exp_peak]) - left_ips))
    elif asym == 'right':
        thres = x[exp_peak] + scale_factor * (2 * (right_ips - x[exp_peak]))
    else:
        thres = x[exp_peak] + scale_factor * width
    
    return thres, x[exp_peak], width, float(height), left_ips, right_ips


def logFWHM(marker, in_data, asym='sym', scale_factor=1):
    
    '''Finds peak + FWHM threshold in a given signal that is first log1p-transformed;
    peak is expected to be highest and leftmost -> if not -> warning message and leftmost peak is used.
    Suitable for left-skewed, zero-bounded data.
    Asym argument can either be None, left, or right depending on assumptions about data distribution.'''
    
    
    data = np.log2(in_data + 1)
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), np.array(data).size)
    y = kde.evaluate(x)
    
    peaks, properties = find_peaks(y, prominence=0.1) #prominence threshold to exclude very small peaks
    print(f'Peaks for {marker} are: {x[peaks]}.')
    print('Prominences: ', properties["prominences"])
    
    exp_peak = peaks[0]
    for peak in peaks:
        if kde(x[peak]) > 3*kde(x[exp_peak]):
            print("Warning: Leftmost and highest peak are not identical! Highest is used.")
            exp_peak = peak
    
    #Assumption: Highest peak corresponds to negative population
    width, height, left_ips, right_ips = peak_widths(y, np.array([exp_peak]), rel_height=0.5)
    width = round(width[0]) * ((np.max(x) - np.min(x)) / x.size)
    left_ips = x[round(left_ips[0])]
    right_ips = x[round(right_ips[0])]
    print('width: ', width, 'left_ips: ', left_ips, 'right_ips: ', right_ips, 'peak: ', x[exp_peak])
    
    if asym == 'left':
        thres = 2 ** (x[exp_peak] + scale_factor * (2 * (x[exp_peak] - left_ips))) - 1
    elif asym == 'right':
        thres = 2 ** (x[exp_peak] + scale_factor * (2 * (right_ips - x[exp_peak]))) - 1
    else:
        thres = 2 ** (x[exp_peak] + scale_factor * width) - 1
    
    return thres, x[exp_peak], width, float(height), left_ips, right_ips

        
def scale(i, idents=None, linearize=False, reverse=False, raw=None):
    
    if linearize:
        if (idents == None).all():
            idents = range(0, np.array(i).size, 1)
            
        rang = list(np.linspace(0, 1, np.array(i).size))
        combined = zip(list(i), list(idents))
        sortlist = sorted(combined)
        combined_2 = zip(sortlist, rang)
        sortlist_2 = sorted(combined_2, key = lambda y: y[0][1])
        scaled_out = [element for _, element in sortlist_2]
        
    elif reverse == True and raw is not None:
        scaled_out = i * (np.max(raw) - np.min(raw)) + np.min(raw)
            
    elif reverse == False and raw is not None:
        if np.min(raw) < 0:
            scaled_out = (i + abs(np.min(raw))) / (np.max(raw) - np.min(raw))
        elif np.min(raw) == 0:
            scaled_out = i / np.max(raw)
        else:
            scaled_out = (i - np.min(raw)) / (np.max(raw) - np.min(raw))
            
    else: #reverse must be False
        if np.min(i) < 0:
            scaled_out = (i + abs(np.min(i))) / (np.max(i) - np.min(i))
        elif np.min(i) == 0:
            scaled_out = i / np.max(i)
        else:
            scaled_out = (i - np.min(i)) / (np.max(i) - np.min(i))
            
    return scaled_out


def negative_exponential(x, b, k):
    return b * (1 - np.exp(-k * x))

def negative_exponential_jacobian(x, b, k):
    return np.array([1 - np.exp(-k * x), b * x * np.exp(-k * x)]).T


def combined_thresholding(marker_dict, markers_use, data, skewness_factors, aberrant_scale, measurement_x:str, plot_dir:str, 
                          measurement_y='prop_50', der_mid=0.1, graphics_save=True, graphics_show=True):
    
    '''
    Write documentation
    '''
    
    thres_intens = {}
    exp_params1 = {}
    exp_params2 = {}
    thres_skew = {}
    thres_ent = {}

    for marker in markers_use:
        
        xdata_raw = data['_'.join([measurement_x, marker_dict[marker]])]
        xdata = scale(data['_'.join([measurement_x, marker_dict[marker]])])
        ydata = scale(data['_'.join([measurement_y, marker_dict[marker]])])
        ydata_raw = data['_'.join([measurement_y, marker_dict[marker]])]

        #fitting (done on scaled data)
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

        #Intensity ranks
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

        try:
            skew_thres, peak_skew, _2, _3, _4, _5 = FWHM(marker=marker, data=skew_data, scale_factor=skewness_factors[marker])
        except Exception as ex:
            print(ex, "Skewness thresholding did not work, set manually!")
            skew_thres = 1


        #graphics
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

        #Load updated data
        data_new = data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, ]
        xdata_new = scale(data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_x, marker_dict[marker]])])
        ydata_new = scale(data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_y, marker_dict[marker]])])
        xdata_raw_new = data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_x, marker_dict[marker]])]
        ydata_raw_new = data.loc[data['_'.join(['skewness', marker_dict[marker]])] < skew_thres, '_'.join([measurement_y, marker_dict[marker]])]

        #Fitting 2nd iteration
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

        #Intensity ranks
        data_new['_'.join(['intens_rank', marker_dict[marker]])] = data_new['_'.join([measurement_x, marker_dict[marker]])].rank(ascending=False, method='first')

        #Data selection for thresholding
        if len(data_new) > 600:
            if data_new.loc[data_new['_'.join(['intens_rank', marker_dict[marker]])] == 600.0, '_'.join([measurement_x, marker_dict[marker]])].to_list()[0] > thres_mid:
                prop_data = data_new.loc[(data_new['_'.join(['intens_rank', marker_dict[marker]])] <= 600), '_'.join(['prop_50', marker_dict[marker]])]
                print("Using first 600 cells for prop50 filtering.")
            else:
                prop_data = data_new.loc[data_new['_'.join([measurement_x, marker_dict[marker]])] > thres_mid, '_'.join(['prop_50', marker_dict[marker]])]
        else:
            prop_data = data_new.loc[data_new['_'.join([measurement_x, marker_dict[marker]])] > thres_mid, '_'.join(['prop_50', marker_dict[marker]])]


        #Dealing with special case where no background is present is not done any more - too arbitrary 
        #Has to fixed manually with GUI now

        #Prop50 threshold (Scale factor is constant of 1) with error handling
        try:
            prop_thres, peak_prop, _2, _3, _4, _5 = FWHM(marker=marker, data=prop_data, asym='right', scale_factor=0.9)
        except Exception as ex:
            print(ex, "Prop50 thresholding did not work, set manually!")
            prop_thres = 0.5

        #Set threholds
        thres_intens[marker] = [thres_incl, 2.5*thres_incl]
        thres_skew[marker] = skew_thres
        thres_ent[marker] = [prop_thres, scale(i = popt[0], reverse=True, raw=ydata_raw_new)]
        exp_params2[marker] = popt


        #graphics
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
    
    '''Write documentation'''
    
    thres_FWHM = {}
    
    for marker in markers_FWHM:
        try:
            thres_marker, _1,_2,_3,_4,_5 = logFWHM(marker, data['_'.join([intensity_measure, marker_dict[marker]])], asym=base_symmetry, scale_factor=base_scale_factor)
        except Exception as ex:
            thres_FWHM[marker] = 1
            print(ex, "FWHM thresholding didn't work! Set manually.")
        else:
            thres_FWHM[marker] = thres_marker
 
    return thres_FWHM


def autofluo_thresholding(data, columns_autofluo, base_scale_factor=1, base_symmetry='sym'): #base_symmetry can be sym, left or right
    
    '''Write documentation'''
    
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

    '''Write documentation.
    Output is list of raw identity strings that can be added to dataframe as new column.
    prop_thres_strictness can be either of high, mid, low'''
    
    raw_ident_list = []
    scale_data_dict = dict(zip(data['unique_ident'], scale(data['intensity'])))
      
        
    for cell in data['unique_ident']: #iterate through all cells
            
        #Gather data
        intens = data.loc[data['unique_ident'] == cell, 'intensity'].to_list()[0]
        prop = data.loc[data['unique_ident'] == cell, 'prop50'].to_list()[0]
        skew = data.loc[data['unique_ident'] == cell, 'skewness'].to_list()[0]
            
        #Prepare dynamic prop50 threshold
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
        
        #Append temp. ID
        if (prop < prop50) & (intens > thres_intens[0]) & (skew < thres_skew):
             raw_ident_list.append('pos')
        else:
            raw_ident_list.append('neg')
    
    return raw_ident_list


def tempo_idents_FWHM(data, thres_FWHM): 

    '''Temp. ID assignment for FWHM thresholded markers.'''
    
    raw_ident_list = []      
        
    for cell in data['unique_ident']:
        intens = data.loc[data['unique_ident'] == cell, 'intensity'].to_list()[0]
        if intens > thres_FWHM:
            raw_ident_list.append('pos')
        else:
            raw_ident_list.append('neg')

    return raw_ident_list


def raw_idents(data, marker_dict, thres_intens, thres_skew, thres_ent, exp_params, thres_FWHM, prop_thres_strictness): 

    '''Write documentation.
    Output is list of raw identity strings that can be added to dataframe as new column.
    prop_thres_strictness can be either of high, mid, low'''
    
    raw_ident_list = []
    ident_list = []
    scale_data_dict = {}
    
    for m in marker_dict.keys():
        scale_data_dict[m] = dict(zip(data['unique_ident'], scale(data['_'.join([intensity_measure, marker_dict[m]])])))
      
    for cell in data['unique_ident']: #iterate through all cells
        
        temp_pos_markers = []

        #Negative exponential
        for marker, thresholds in thres_intens.items():
            
            #Gather data
            intens = data.loc[data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict[marker]])].to_list()[0]
            prop = data.loc[data['unique_ident'] == cell, '_'.join(['prop_50', marker_dict[marker]])].to_list()[0]
            skew = data.loc[data['unique_ident'] == cell, '_'.join(['skewness', marker_dict[marker]])].to_list()[0]
            
            #Prepare dynamic prop50 threshold
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
                
            if (prop < prop50) & (intens > thresholds[0]) & (skew < thres_skew[marker]):
                temp_pos_markers.append(marker)

        #FWHM
        for marker, thresholds in thres_FWHM.items():
            intens = data.loc[data['unique_ident'] == cell, '_'.join([intensity_measure, marker_dict[marker]])].to_list()[0]
            if intens > thresholds:
                temp_pos_markers.append(marker)

        
        #Cells that are not positive for any marker
        if len(temp_pos_markers) == 0:
            raw_ident_list.append('undefined')

        #Cells that are positive for at least one marker   
        elif len(temp_pos_markers) == 1:
            raw_ident_list.append(temp_pos_markers[0])
        else:
            raw_ident_list.append('_'.join(temp_pos_markers))

    
    return raw_ident_list


#Helper func for power set, excl {}, subsets added conditionally
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


def power_set_alt(items, conds):
    N = len(items)
    out_sets = []
    # enumerate the 2 ** N possible combinations
    for i in range(2 ** N):
        combo = set([])
        for j in range(N):
            # test bit jth of integer i
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
    
    '''window for check of FWHM thresholded markers'''
    
    def __init__(self, in_data, thres_intens, FOV_ranks, FOV_dict, seg_dict, marker, *args, **kwargs):
        
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
        
        #DYNAMIC VARS
        self.intVar = DoubleVar()
        self.intVar.set(self.thres_intens)
        
        self.brightVar = DoubleVar()
        self.brightVar.set(2)
        
        self.currFOV = StringVar()
        self.currFOV.set(self.FOV_ranks[0])
        
        self.intensity = np.array(self.in_data['intensity'])
        self.intensity = self.intensity[~np.isnan(self.intensity)]
        
        #PRESETTING TRACKING VARS
        self.track_int = self.intVar.get()
        self.track_bright =  self.brightVar.get()
        self.track_currFOV = self.currFOV.get()
        
        #WIDGETS
    
        #TOP ROW
        #Reset and refresh buttons
        self.refreshbutton = ttk.Button(master=self.saveframe, text='Refresh', style='save.TButton', command=self.update_window)
        self.refreshbutton.pack()
        self.resetbutton = ttk.Button(master=self.saveframe, text='Reset', style='save.TButton', command=self.reset_thres)
        self.resetbutton.pack()
        #Save button to accept thresholds and exit GUI
        self.buttonStyle = ttk.Style()
        self.buttonStyle.configure('save.TButton', width=80)
        #self.savebutton = ttk.Button(master=self.saveframe, text='Save', style='save.TButton', command=self.quit_save)
        #self.savebutton.pack()

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
        self.labInt = ttk.Label(self.setframe12, text='Intensity')
        self.labInt.pack()
        self.scaleInt = ttk.Scale(self.setframe12, orient=HORIZONTAL, length=350, from_=min(self.intensity), to=max(self.intensity), variable=self.intVar)
        self.scaleInt.pack()
        self.spinInt = ttk.Spinbox(self.setframe12, from_=min(self.intensity), to=max(self.intensity), textvariable=self.intVar, increment=1)
        self.spinInt.pack()

        self.labBright = ttk.Label(self.dropframe2, text='Brightness')
        self.labBright.pack()
        self.scaleBright = ttk.Spinbox(self.dropframe2, from_=0, to=30, increment=0.1, textvariable=self.brightVar)
        self.scaleBright.pack()
        
        #IMAGES/GRAPHS
        #Create first version of column 'preID'
        self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
        self.in_data_sel['preID'] = tempo_idents_FWHM(self.in_data_sel, thres_FWHM = self.intVar.get())

        #Loading first qsub and seg images
        self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
        self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')

        #IMAGES
        self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
        self.im_arr = self.im_arr.astype(np.uint8)
        self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])
        
        self.im2 = Img.fromarray(self.im_arr)
        self.im2en = ImageEnhance.Brightness(self.im2)
        self.im2 = self.im2en.enhance(self.brightVar.get())  #Baseline brightness scaling factor of 2
        #self.im2 = self.im2en.resize((self.im2en.width//3, self.im2en.height//3))

        self.python_im2 = ImageTk.PhotoImage(self.im2, master=self.imageframe2)
        self.mid_im = Label(self.imageframe2,  image=self.python_im2)
        self.mid_im.pack()
        #mid image placed
        
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
        #self.Hm_rgb = self.Hm_rgb.resize((self.Hm_rgb.width//3, self.Hm_rgb.height//3))

        self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb, master=self.imageframe3)
        self.right_im = Label(self.imageframe3,  image=self.python_im3)
        self.right_im.pack()
        #right image placed

        #Create and place plot

        self.fig = Figure(figsize = (4, 4), dpi = 100)
        self.plot = self.fig.add_subplot(111)
        self.plot.hist(self.in_data_sel['intensity'], bins=50)
        self.plot.axvline(x=self.intVar.get(), c='red')

        # creating and placing the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, self.imageframe1)  
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # creating and placing the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.imageframe1)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()
        
        #Initalization finished
        
        
    #Stop updating thread and close window
    def window_del(self):
        global intens_thres 
        intens_thres = self.intVar.get()
        self.destroy()
    
    #Write to global vars and close window
    #def quit_save(self):
    #    global intens_thres 
    #    intens_thres = self.intVar.get()
    #    self.destroy()
                     
    #Reset all thresholds to baseline
    def reset_thres(self):
        self.intVar.set(self.thres_intens)
        
    #Updating method that starts seperate thread
    def update_window(self):
        #self.update_thread = threading.Thread(target=self.reload_win).start()
        self.reload_win()
        
    #Method that is being outsourced to seperate thread
    def reload_win(self):
    
        #Set status
        self.status.config(text='Wait', foreground='#ff0000')
        
        self.IDnew = False
        self.FOVnew = False
        
        if self.track_currFOV != self.currFOV.get():
            
            self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
            
            self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
            self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')
            
            self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
            self.im_arr = self.im_arr.astype(np.uint8)
            self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])

            self.im2 = Img.fromarray(self.im_arr)
            self.im2en = ImageEnhance.Brightness(self.im2)
            self.im2 = self.im2en.enhance(self.brightVar.get())  #Baseline brightness scaling factor of 2
            #self.im2 = self.im2en.resize((self.im2en.width//3, self.im2en.height//3))

            self.python_im2 = ImageTk.PhotoImage(self.im2)
            self.mid_im.config(image=self.python_im2)
            self.mid_im.image = self.python_im2
            #mid image placed
            self.FOVnew = True
            
        elif self.FOVnew == False and self.track_bright != self.brightVar.get():
            
            self.im2 = Img.fromarray(self.im_arr)
            self.im2en = ImageEnhance.Brightness(self.im2)
            self.im2 = self.im2en.enhance(self.brightVar.get())  #Baseline brightness scaling factor of 2
            #self.im2 = self.im2en.resize((self.im2en.width//3, self.im2en.height//3))

            self.python_im2 = ImageTk.PhotoImage(self.im2)
            self.mid_im.config(image=self.python_im2)
            self.mid_im.image = self.python_im2
            self.brightnew = True
            
        if self.track_int != self.intVar.get() or self.track_currFOV != self.currFOV.get():
            
            #Recalculating IDs
            self.in_data_sel['preID'] = tempo_idents_FWHM(self.in_data_sel, thres_FWHM = self.intVar.get())
            self.IDnew = True
            
        if self.FOVnew == True or self.IDnew == True:
            
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
            #self.Hm_rgb = self.Hm_rgb.resize((self.Hm_rgb.width//3, self.Hm_rgb.height//3))
            self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
            self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

            self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
            self.right_im.config(image=self.python_im3)
            self.right_im.image = self.python_im3
            #right image placed

            #Update plot
            self.fig.clear()
            self.plot = self.fig.add_subplot(111)
            self.plot.hist(self.in_data_sel['intensity'], bins=50)
            self.plot.axvline(x=self.intVar.get(), c='red')

            self.canvas.draw_idle()
            self.toolbar.update()
            
        elif self.FOVnew == False and self.IDnew == False and self.brightnew == True:
            
            self.im_arr2 = self.im_arr.copy()
            self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
            self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
            self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
            self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

            self.Hm_rgb = Img.fromarray(self.im_arr2)
            #self.Hm_rgb = self.Hm_rgb.resize((self.Hm_rgb.width//3, self.Hm_rgb.height//3))
            self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
            self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

            self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
            self.right_im.config(image=self.python_im3)
            self.right_im.image = self.python_im3
            #right image placed
            
            
        self.status.config(text='Ready', foreground='#009933')
        
        #Resetting tracking vars
        self.track_int = self.intVar.get()
        self.track_currFOV = self.currFOV.get()
        self.track_bright = self.brightVar.get()
        
    

class GUI_exp(Tk): 
    
    '''Args/kwargs should be in_data,
       thres_intens, thres_skew, thres_ent, exp_params, prop_thres_strictness,
       FOV_ranks:list, FOV_dict:dict, seg_dict:dict
       window for check of neg. exp. thresholded markers
       entries on FOV_ranks == keys of FOV_dict and seg_dict.
       in_data must have: intensity, skewness, prop50, unique_ident, label, FOV.
       preID is created first to work with further along the function.'''
    
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
        
        #DYNAMIC VARS
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
    
        
        #PRESETTING TRACKING VARS
        self.track_int = self.intVar.get()
        self.track_skew = self.skewVar.get()
        self.track_prop = self.propVar.get()
        self.track_bright =  self.brightVar.get()
        self.track_currFOV = self.currFOV.get()
        
        #WIDGETS
    
        #TOP ROW
        #Reset and refresh buttons
        self.refreshbutton = ttk.Button(master=self.saveframe, text='Refresh', style='save.TButton', command=self.update_window)
        self.refreshbutton.pack()
        self.resetbutton = ttk.Button(master=self.saveframe, text='Reset', style='save.TButton', command=self.reset_thres)
        self.resetbutton.pack()
        #Save button to accept thresholds and exit GUI
        self.buttonStyle = ttk.Style()
        self.buttonStyle.configure('save.TButton', width=80)
        #self.savebutton = ttk.Button(master=self.saveframe, text='Save', style='save.TButton', command=self.quit_save)
        #self.savebutton.pack()

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
            
        self.labBright = ttk.Label(self.dropframe2, text='Brightness')
        self.labBright.pack()
        self.scaleBright = ttk.Spinbox(self.dropframe2, from_=0, to=30, increment=0.1, textvariable=self.brightVar)
        self.scaleBright.pack()
        
        #IMAGES/GRAPHS
        #Create first version of column 'preID'
        self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
        self.in_data_sel['preID'] = tempo_idents_exp(self.in_data_sel, 
                                            thres_intens = [self.intVar.get(), self.intVar.get()*2.5], 
                                            thres_skew = self.skewVar.get(), 
                                            thres_ent = [self.thres_prop[1]+self.propVar.get(), self.thres_prop[1]], 
                                            exp_params = self.exp_params, 
                                            prop_thres_strictness = self.prop_thres_strictness)

        #Loading first qsub and seg images
        self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
        self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')

        #IMAGES
        self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
        self.im_arr = self.im_arr.astype(np.uint8)
        self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])
        
        self.im2 = Img.fromarray(self.im_arr)
        self.im2en = ImageEnhance.Brightness(self.im2)
        self.im2 = self.im2en.enhance(self.brightVar.get())  #Baseline brightness scaling factor of 2
        #self.im2 = self.im2en.resize((self.im2en.width//3, self.im2en.height//3))

        self.python_im2 = ImageTk.PhotoImage(self.im2, master=self.imageframe2)
        self.mid_im = Label(self.imageframe2,  image=self.python_im2)
        self.mid_im.pack()
        #mid image placed
        
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
        #self.Hm_rgb = self.Hm_rgb.resize((self.Hm_rgb.width//3, self.Hm_rgb.height//3))

        self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb, master=self.imageframe3)
        self.right_im = Label(self.imageframe3,  image=self.python_im3)
        self.right_im.pack()
        #right image placed

        #Create and place plot

        self.fig = Figure(figsize = (4, 4), dpi = 100)
        self.plot = self.fig.add_subplot(111)
        self.plot.scatter(x = self.in_data_sel['intensity'], 
                          y = self.in_data_sel['prop50'],
                          c = list(map(posneg, self.in_data_sel['preID'])),
                          alpha = 0.5)

        # creating and placing the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, self.imageframe1)  
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # creating and placing the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.imageframe1)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()
        
        #Initalization finished
        
        
    #Stop updating thread and close window
    def window_del(self):
        global intens_thres 
        intens_thres = [self.intVar.get(), self.intVar.get()*2.5]
        global skew_thres 
        skew_thres = self.skewVar.get()
        global ent_thres 
        ent_thres = [self.thres_prop[1]+self.propVar.get(), self.thres_prop[1]]
        self.destroy()
    
    #Write to global vars and close window
    #def quit_save(self):
        
        #self.destroy()
                     
    #Reset all thresholds to baseline
    def reset_thres(self):
        self.intVar.set(self.thres_intens[0])
        self.skewVar.set(self.thres_skew)
        self.propVar.set(self.thres_prop[0] - self.thres_prop[1])
        
    #Updating method that starts seperate thread
    def update_window(self):
        #self.update_thread = threading.Thread(target=self.reload_win).start()
        self.reload_win()
        
    #Method that is being outsourced to seperate thread
    def reload_win(self):
    
        #Set status
        self.status.config(text='Wait', foreground='#ff0000')
        
        self.IDnew = False
        self.FOVnew = False
        
        if self.track_currFOV != self.currFOV.get():
            
            self.in_data_sel = self.in_data.loc[self.in_data['FOV'] == self.currFOV.get(), : ]
            
            self.start_qsub = imread(self.FOV_dict[self.currFOV.get()], plugin='pil')
            self.start_seg = imread(self.seg_dict[self.currFOV.get()], plugin='pil')
            
            self.im_arr = (self.start_qsub/np.max(self.start_qsub))*255
            self.im_arr = self.im_arr.astype(np.uint8)
            self.im_arr = np.dstack([self.im_arr, self.im_arr, self.im_arr])

            self.im2 = Img.fromarray(self.im_arr)
            self.im2en = ImageEnhance.Brightness(self.im2)
            self.im2 = self.im2en.enhance(self.brightVar.get())  #Baseline brightness scaling factor of 2
            #self.im2 = self.im2en.resize((self.im2en.width//3, self.im2en.height//3))

            self.python_im2 = ImageTk.PhotoImage(self.im2)
            self.mid_im.config(image=self.python_im2)
            self.mid_im.image = self.python_im2
            #mid image placed
            self.FOVnew = True
            
        elif self.FOVnew == False and self.track_bright != self.brightVar.get():
  
            self.im2 = Img.fromarray(self.im_arr)
            self.im2en = ImageEnhance.Brightness(self.im2)
            self.im2 = self.im2en.enhance(self.brightVar.get())  #Baseline brightness scaling factor of 2
            #self.im2 = self.im2en.resize((self.im2en.width//3, self.im2en.height//3))

            self.python_im2 = ImageTk.PhotoImage(self.im2)
            self.mid_im.config(image=self.python_im2)
            self.mid_im.image = self.python_im2
            self.brightnew = True
            
        if self.track_int != self.intVar.get() or self.track_skew != self.skewVar.get() or self.track_prop != self.propVar.get() or self.track_currFOV != self.currFOV.get():
            
            #Recalculating IDs
            self.in_data_sel['preID'] = tempo_idents_exp(self.in_data_sel, 
                                                     thres_intens = [self.intVar.get(), self.intVar.get()*2.5], 
                                                     thres_skew = self.skewVar.get(), 
                                                     thres_ent = [self.thres_prop[1]+self.propVar.get(), self.thres_prop[1]], 
                                                     exp_params = self.exp_params, 
                                                     prop_thres_strictness = self.prop_thres_strictness)
            
            self.IDnew = True
            
        if self.FOVnew == True or self.IDnew == True:
            
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
            #self.Hm_rgb = self.Hm_rgb.resize((self.Hm_rgb.width//3, self.Hm_rgb.height//3))
            self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
            self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

            self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
            self.right_im.config(image=self.python_im3)
            self.right_im.image = self.python_im3
            #right image placed

            #Update plot
            self.fig.clear()
            self.plot = self.fig.add_subplot(111)
            self.plot.scatter(x = self.in_data_sel['intensity'], 
                              y = self.in_data_sel['prop50'],
                              c = list(map(posneg, self.in_data_sel['preID'])),
                              alpha = 0.5)

            self.canvas.draw_idle()
            self.toolbar.update()
            
            
        elif self.FOVnew == False and self.IDnew == False and self.brightnew == True:
            
            self.im_arr2 = self.im_arr.copy()
            self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 0:2] = 0
            self.im_arr2[np.where(self.one_hot_hm==1)[0], np.where(self.one_hot_hm==1)[1], 2] = 255
            self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 0] = 255
            self.im_arr2[np.where(self.one_hot_hm==2)[0], np.where(self.one_hot_hm==2)[1], 1:3] = 0

            self.Hm_rgb = Img.fromarray(self.im_arr2)
            #self.Hm_rgb = self.Hm_rgb.resize((self.Hm_rgb.width//3, self.Hm_rgb.height//3))
            self.Hm_rgb = ImageEnhance.Brightness(self.Hm_rgb)
            self.Hm_rgb = self.Hm_rgb.enhance(self.brightVar.get()) 

            self.python_im3 = ImageTk.PhotoImage(self.Hm_rgb)
            self.right_im.config(image=self.python_im3)
            self.right_im.image = self.python_im3
            #right image placed
            
        self.status.config(text='Ready', foreground='#009933')
        
        #Resetting tracking vars
        self.track_int = self.intVar.get()
        self.track_skew = self.skewVar.get()
        self.track_prop = self.propVar.get()  
        self.track_currFOV = self.currFOV.get()
        self.track_bright = self.brightVar.get()