'''
Functions for part 2 of main HNSCC image analysis pipeline: Alignment, quench subtraction, segmentation and data extraction
authors: Jan Hoelzl, Hannah Peterson

Center for Systems Biology
Massachusetts General Hospital
'''

##FUNCS

def euc_align(im1,im2):
    
    # Read the images to be aligned
    # Convert images to grayscale
    if len(im1.shape) != 2:
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = im1
        im2_gray = im2
    # Find size of image1
    sz = im1.shape
    
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN #cv2.MOTION_TRANSLATION
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Specify the number of iterations.
    number_of_iterations = 5000;
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None,15)
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix


def color2gray(im_col,lum=True):
    if lum == True:
        im_gray = im_col[:,:,0]*0.3 + im_col[:,:,1]*0.59 + im_col[:,:,2]*0.11
    if lum == False: 
        im_gray = im_col[:,:,0]/3. + im_col[:,:,1]/3. + im_col[:,:,2]/3.
    return im_gray


def data_prep_nuc_cell_match(file_mask_nuc,file_mask_cell,graphics=False):
    '''
    file_im_raw :: str, filepath to image file
    file_mask_nuc :: str, filepath to nuclei mask file
    file_mask_cell :: str, filepath to cell mask file
    graphics :: bool, print graphics of segmentation
    
    mask_nuc :: array, mask of nuclei for analysis
    mask_cell :: array, mask of cells for analysis
    props :: DataFrame, matching nuclei and cell mask labels with corresponding values
    len(droplist) :: int, number of rejected objects from analysis
    '''
    # images
    #im_raw = skimage.io.imread(file_im_raw,plugin='pil')
    if type(file_mask_nuc) == str:
        mask_nuc_raw = skimage.io.imread(file_mask_nuc,plugin='pil')
        mask_cell_raw = skimage.io.imread(file_mask_cell,plugin='pil')
    else:
        mask_nuc_raw = file_mask_nuc
        mask_cell_raw = file_mask_cell
    
    # data for analysis
    # match cells and nuclei
    matching = {}
    for n in np.unique(mask_nuc_raw.flatten()):
        n = int(n)
        matching[n] = {}
        matching[n]['cyto_match'] = np.argmax(np.bincount(list(mask_cell_raw[mask_nuc_raw==n]))) # find most frequent cyto_mask number within nuc_mask
        matching[n]['nuc_area'] = sum(sum(np.ma.masked_where(mask_nuc_raw==n,mask_nuc_raw).mask)) # nucleus pixel area
        matching[n]['cyto_area'] = sum(sum(np.ma.masked_where(mask_cell_raw==matching[n]['cyto_match'],mask_cell_raw).mask)) # cytoplasm pixel area   
        #matching[n]['cyto_int_sum'] = sum(im_raw[:,:][np.ma.masked_where(mask_cell_raw==matching[n]['cyto_match'],mask_cell_raw).mask]) # sum of cell intensity
        matching[n]['ncr'] = matching[n]['nuc_area']/matching[n]['cyto_area']
    df_match = pd.DataFrame(matching).T

    # remove nuclei without cells and cells with 2 nuclei and those whose NCR>1 and cells with no nuclei
    droplist_n = list(df_match[df_match.ncr>1.00].index) + list(df_match[df_match.cyto_match==0].index) + list(df_match[df_match.cyto_match.duplicated(keep=False)].index) 
    droplist_c = list(set(mask_cell_raw.flatten())-set(df_match.cyto_match))

    mask_nuc = mask_nuc_raw.copy()
    mask_cell = mask_cell_raw.copy()
    
    for c in droplist_n:
        mask_nuc[np.ma.masked_where(mask_nuc==c,mask_nuc).mask] = 0 # make dropped nuclei into background
        mask_cell[np.ma.masked_where(mask_cell==df_match.cyto_match[c],mask_cell).mask] = 0 # make dropped cell into background
    for c in droplist_c:
        mask_cell[np.ma.masked_where(mask_cell==c,mask_cell).mask] = 0 # make dropped cell into background
        
    cells = df_match.drop(droplist_n)
    cells['nuc_match'] = cells.index
    
    relabel_nucs = np.zeros((mask_nuc.shape))
    for nuc in cells['nuc_match']:
        relabel_nucs[np.where(mask_nuc == int(nuc))] = int(cells.loc[cells['nuc_match'] == nuc, 'cyto_match'].to_list()[0])

    # graphics
    if graphics==True:
        # segmented files
        plt.imshow(skimage.color.label2rgb(mask_nuc_raw,bg_label=0))
        plt.imshow(skimage.color.label2rgb(mask_cell_raw,bg_label=0))
        # matched cells and nuclei for analysis
        plt.imshow(skimage.color.label2rgb(mask_nuc,bg_label=0))
        plt.imshow(skimage.color.label2rgb(mask_cell,bg_label=0))
    
    return mask_nuc, mask_cell, relabel_nucs, len(droplist_n+droplist_c)


def mask_heatmap(mask,props,identity): #Old function only included here for completeness, use updated one in Pipeline_part2
    '''
    mask :: array, assumes mask background is 0
    props :: DataFrame, must have mask labels and column 'label'
    identity :: str, must be a column of values in dataframe props

    int_mask :: array of values, same shape as mask
    '''
    assert identity in props.columns

    int_mask = mask.copy()
    for c in np.unique(mask.flatten()):
        if c != 0:
            int_mask[np.ma.masked_where(mask==c,mask).mask] = props[props['label']==c][identity].values[0]

    return int_mask


def IQMean(mask, qsub, qL: float, qU: float):

    """
    Calculates the mean intensity of objects identified within a mask, bounded by lower and upper quantiles.

    Parameters
    ----------
    mask : array
           segmentation mask
    qsub : array
           quench subtracted image
    qL : float
        Lower quantile boundary for intensity measurements (0 < qL < 1).
    qU : float
        Upper quantile boundary for intensity measurements (qL < qU <= 1).

    Returns
    -------
    IQmean_intensities : list
                         A list of mean intensities for each cell in the segmentation mask, truncated between the qL and qU quantiles.

    Notes
    -----
    This function assumes that the mask's background is represented by 0 and objects are identified by unique non-zero values.
    """

    regions = np.unique(mask.flatten())[1:]
    IQmean_intensities = []
    
    for region in regions:

        obj = qsub[np.where(mask == region)].flatten()
        obj.sort(kind='stable')

        xKupper = math.ceil(qU * obj.size)
        xKlower = math.floor(qL * obj.size)
        
        truncated_obj = obj[xKlower:xKupper]
        IQmean_intensities.append(np.mean(truncated_obj))
    
    return IQmean_intensities


def skewness_of_stain(weighted_centroids_x, weighted_centroids_y,
                      centroids_x, centroids_y, equivalent_diameter):
    
    '''
    Calculates skewness of stain metric
    weighted_centroids_x, weighted_centroids_y, centroids_x, centroids_y , equivalent_diameter :: 1d arrays
    '''
    
    #load data into arrays
    weighted_centroids = np.vstack((weighted_centroids_x, weighted_centroids_y)).T
    centroids = np.vstack((centroids_x, centroids_y)).T
    
    out = np.array([])
    for cell in range(0, equivalent_diameter.size, 1):
        dist = np.linalg.norm((weighted_centroids[cell, :] - centroids[cell, :]))
        norm_dist = dist / (equivalent_diameter[cell] / 2)
        out = np.append(out, norm_dist)
    
    return out
        

def centrality_of_stain(images, intensity_images):
    
    '''
    Calculates the centrality of stain metric (not used currently but included just in case)
    images, intensity_images :: Dicts containing arrays of binary and intensity images for all objects
    '''
    
    out_ratio = np.array([])
    out_tendency = np.array([])
    out_spearman = np.array([])
    
    for cell in range(0, images.size, 1):
        mask = images[cell].astype(int)
        image = intensity_images[cell]
        means = np.array([])

        while True:
            erode_1 = scipy.ndimage.morphology.binary_erosion(mask, iterations = 1)
            measure = mask - erode_1
            temp_mean = np.mean(image[measure.astype(np.bool)])
            if np.isnan(temp_mean):
                break
            means = np.append(means, temp_mean)
            mask = erode_1.astype(int)
        
        value_space1 = np.linspace(1, -1, num=means.size)
        weighted_means1 = means * value_space1
        out_ratio = np.append(out_ratio, np.mean(weighted_means1))
        
        value_space2 = np.linspace(1, 0, num=means.size)
        tendency = np.sum(value_space2 * (means / np.sum(means)))
        out_tendency = np.append(out_tendency, tendency)
        
        value_space3 = np.arange(1, (means.size + 1), 1)
        spearman_corr = scipy.stats.spearmanr(value_space3, means)[0]
        out_spearman = np.append(out_spearman, - spearman_corr)
    
    return out_ratio, out_tendency, out_spearman


def prop_ness(images, intensity_images, prop):
    
    '''Calculates proportion of pixels necessary for 50% of the cell of interest's total stain'''
    
    out = np.array([])
    
    for cell in range(0, images.size, 1):
        mask = images[cell].astype(np.bool)
        image = intensity_images[cell]
        measure = image[mask].flatten()
        measure = np.sort(measure, kind='stable')[::-1]
        arr_sum = np.sum(measure) * prop
        
        count_sum = 0
        counter = 0
        for element in measure:
            if count_sum < arr_sum:
                counter += 1
                count_sum += element
            else:
                break
        prop_needed = counter / measure.size
        out = np.append(out, prop_needed)
        
    return out


def mutual_information_2d(x, y, normalized=True):
    
    """
    Computes (normalized) mutual information (MI) between two 1D variables from a
    joint histogram.

    Parameters
    ----------
    x : 1D array -> use array.ravel() for greyscale images
        first variable
    y : 1D array -> use array.ravel() for greyscale images
        second variable
    normalized : Boolean (defaults to True)
                 Decides whether the calculated MI is to be normalized to [0,1]

    Returns
    -------
    mi: float
        Mutual information as a similarity measure
    """
    
    nbins=256 #Number of bins to use for histogram; can be changed depending on desired resolution of MI calculation
    EPS = np.finfo(float).eps
    bins = (nbins, nbins)
    jh = np.histogram2d(x, y, bins=bins)[0]

    # Smooth the joint histogram with a gaussian filter of sigma=1 for slight noise reduction
    ndimage.gaussian_filter(jh, sigma=1, mode='constant', output=jh)

    # Compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2)))

    return mi


def MI_entropy(im1, im2):
    
    x, y = im1.shape
    num_tilesx = im1.shape[0]//100
    num_tilesy = im1.shape[1]//100
    step_x = int((x - (x % num_tilesx)) / num_tilesx)
    step_y = int((y - (y % num_tilesy)) / num_tilesy)
    
    MI_results = []
    
    for i in range(1, (num_tilesx + 1)):
        for z in range(1, (num_tilesy + 1)):
            temp1 = im1[((i * step_x) - step_x) : (i * step_x), ((z * step_y) - step_y) : (z * step_y)]
            temp2 = im2[((i * step_x) - step_x) : (i * step_x), ((z * step_y) - step_y) : (z * step_y)]
            MI_results.append((1/mutual_information_2d(temp1.ravel(), temp2.ravel(), sigma=1, normalized=True, nbins=round(np.sqrt((temp1.size/5)))))**2)
            
    entr = scipy.stats.entropy(MI_results, base=len(MI_results))
    
    return entr


def optimizing_func(x, raw, sub):
    '''Function for MI minimization'''
    opt_im = raw - x*sub
    return (mutual_information_2d(sub.ravel(), opt_im.ravel(), sigma=1, normalized=True, nbins=round(np.sqrt((opt_im.size/5)))))


def Minimize_MI(imDict):
    
    '''
    Function to minimize MI between qsub images from cycle 0 to cycle 7.
    imDict :: Dictionary containing image matrices; keys in form "cycleX"
    '''
    cycle_list = sorted(imDict.keys())
    
    corrDict = {}
    corrDict[cycle_list[0]] = imDict[cycle_list[0]]
    outDict = {}
    
    for i in range(1, len(cycle_list)):
        cycle = cycle_list[i]
        Im_raw = imDict[cycle]
        Im_use = cycle_list[:i]
        subDict = {}
        
        for cyc in Im_use:
            Im = corrDict[cyc]
            x_ran = np.arange(0,1,0.02)
            y_ran = np.array([])
            #Evaluate
            for x in x_ran:
                y_ran = np.append(y_ran, optimizing_func(x, raw=Im_raw, sub=Im))
            
            #Fit polynomial of degree 5
            polyfit = np.polynomial.polynomial.Polynomial.fit(x=x_ran, y=y_ran, deg=5)
            fit = polyfit.linspace(n=10000)
            
            x_min = fit[0][np.argmin(fit[1])]
            subDict[cyc] = x_min
 
        for prev, alpha in subDict.items():
            Im_raw = Im_raw - alpha*corrDict[prev]
            
        corrDict[cycle] = np.maximum(0, Im_raw)  
        outDict[cycle] = subDict
    
    return outDict


def Correct_im(imDict, alphas):
    
    '''
    Function to minimize MI between qsub images from cycle 0 to cycle 7.
    imDict :: Dictionary containing image matrices; keys in form "cycleX"
    '''
    cycle_list = sorted(imDict.keys())
    
    corrDict = {}
    corrDict[cycle_list[0]] = imDict[cycle_list[0]]
    
    for i in range(1, len(cycle_list)):
        cycle = cycle_list[i]
        Im_raw = imDict[cycle]
        Im_use = cycle_list[:i]
        subDict = alphas[cycle]
        
        for prev, alpha in subDict.items():
            Im_raw = Im_raw - alpha*corrDict[prev]
            
        corrDict[cycle] = np.maximum(0, Im_raw)
    
    return corrDict


def annotate_artifacts(int_mask, im1, im2, int_thres_global, int_thres_local, ratio_thres, rad):
    
    '''
    int_mask :: cell mask, assumes mask background is 0
    img1 and img2 :: aligned and cropped images (non quench-subtracted!) to be assessed
    thres_global :: Global threshold (int or float)
    thres_local :: Local threshold (int or float)
    rad :: Radius for frequency filtering (int)
    '''
    
    print(np.mean(im1), np.mean(im2))
    mask_unique = np.unique(int_mask)[1:]
    #Denoise with sigma=2
    img1 = ndimage.gaussian_filter(im1.copy(), sigma=2)
    img2 = ndimage.gaussian_filter(im2.copy(), sigma=2)
    
    #Create DFT frequency measurement mask
    measure = np.zeros(img1.shape)
    rr, cc = draw.disk((measure.shape[0]/2, measure.shape[1]/2), radius=rad, shape=measure.shape)
    measure[rr,cc] = 1
    
    #FFT and low/hi frequency amplitude ratio calculation
    FFT1 = scipy.fft.fftshift(scipy.fft.fft2(img1))
    FFT2 = scipy.fft.fftshift(scipy.fft.fft2(img2))
    FFTsub = np.abs(FFT2) - np.abs(FFT1)
    FratioG = np.mean(FFTsub[measure == 1]) / np.maximum(1, np.mean(FFTsub[measure == 0]))
    print(FratioG)
    
    #Global threshold
    if FratioG < ratio_thres or (np.mean(img2) - np.mean(img1)) < int_thres_global:
        #Output is list of strings that are formatted like this: Pass/Fail_GlobalFrequencyFatio_LocalFrequencyRatio (if calculated, otherwise 0)
        #Length of list if cellnumber in FOV -> can be assigned as a column in the measurement df
        outstring = '_'.join(['PassG', str(FratioG), '0'])
        return [outstring]*(mask_unique.size)
    
    else:
        #Recreate the images only using the low frequency spectrum
        #Initialize complex array
        artifact_mask = np.zeros(measure.shape, dtype='complex128')
        #Enter quench-subtracted (in fourier space) low frequenciy values
        artifact_mask[rr,cc] = FFT2[measure == 1] - FFT1[measure == 1]
        #Reverse FFT (shouldn't matter whether np.abs or np.real is used; imaginary part should be zeroed out in all entries)
        recreate_img = np.abs(scipy.fft.ifft2(scipy.fft.fftshift(artifact_mask)))
        
        #Threshold and label 
        #Threshold_mean seems to work best, also tried Otsu but that seems to be to conservative
        thres = threshold_mean(recreate_img)
        recreate_img = recreate_img > thres
        labels = skimage.measure.label(recreate_img, connectivity=2)   

        #Check affected areas
        #Initalize dictionary to contain the FRatio values for all affected areas
        check_dict = {0:0}
        for c in np.unique(labels)[1:]:
            obj_coord = np.where(labels==c)
            if obj_coord[0].size > 2500: #Size threshold of 2500 pixels
                
                #Measure local FRatio (same approach as global)
                #Get subimages
                obj1 = img1[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1]
                obj2 = img2[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1]
                
                print((np.mean(obj2) - np.mean(obj1)))
                if (np.mean(obj2) - np.mean(obj1)) >= int_thres_local:
                    measure = np.zeros(obj1.shape)
                    rr, cc = draw.disk((measure.shape[0]/2, measure.shape[1]/2), radius=rad, shape=measure.shape)
                    measure[rr,cc] = 1

                    FFT1 = scipy.fft.fftshift(scipy.fft.fft2(obj1))
                    FFT2 = scipy.fft.fftshift(scipy.fft.fft2(obj2))
                    FFTsub = np.abs(FFT2) - np.abs(FFT1)
                    FratioL = np.mean(FFTsub[measure == 1]) / np.maximum(1, np.mean(FFTsub[measure == 0]))

                    check_dict[c] = FratioL
        
        #Create output
        out_list = []
        print(np.unique(labels))
        print(check_dict)
            
        for cell in mask_unique:
            regions = np.unique(labels[int_mask == cell])
            rel_keys = sorted(list(check_dict.keys()))[1:]
            cell_ratios = [check_dict[region] for region in regions if region in rel_keys]
            cell_ratios.append(1)
            #If this max. is not unique, I'll actually eat a broom :D
            max_ratio = max(cell_ratios)
            
            if max_ratio < ratio_thres:
                out_list.append('_'.join(['PassL', str(FratioG), str(max_ratio)]))
            else:
                out_list.append('_'.join(['FailL', str(FratioG), str(max_ratio)]))
        
        return out_list

#Create dict with median values


def median_alpha(inDict): 
    
    '''Function to get median alpha values
    in_dict :: dict of dicts containing alpha values'''
    
    outDict = inDict[list(inDict.keys())[0]]
    
    for cycle, subs in outDict.items():
        for cyc, alpha in subs.items():
            collect_array = np.array([])
            for im in inDict.keys():
                collect_array = np.append(collect_array, inDict[im][cycle][cyc])
            outDict[cycle][cyc] = np.median(collect_array)
            
    return outDict

       
#Extra function for regions_props that calculates median intensity
def p50(regionmask,intensity_image):
    return np.quantile(intensity_image[regionmask],0.50)