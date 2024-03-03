'''
Functions for part 2 of main HNSCC image analysis pipeline: Alignment, quench subtraction, segmentation and data extraction
authors: Jan Hoelzl, Hannah Peterson

Center for Systems Biology
Massachusetts General Hospital
'''

##FUNCTIONS

def euc_align(im1, im2):

    """
    Aligns two images (im1 and im2) using the Euclidean transformation approach with Enhanced Correlation Coefficient (ECC) optimization. 
    This method attempts to find the optimal Euclidean transformation (rotation and translation) that aligns im2 to im1.

    Parameters
    ----------
    im1 : array_like
        The reference image to which im2 is aligned. It can be grayscale or BGR (will be converted to grayscale internally).
    im2 : array_like
        The image to be aligned to im1. It can be grayscale or BGR (will be converted to grayscale internally).

    Returns
    -------
    im2_aligned : array_like
        The aligned version of im2.
    warp_matrix : ndarray
        The 2x3 Euclidean transformation matrix.

    Notes
    -----
    - If the motion model is set to homography, a 3x3 transformation matrix is used instead.
    """
    
    #Convert images to grayscale
    if len(im1.shape) != 2:
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = im1
        im2_gray = im2

    #Find size of image1
    sz = im1.shape
    
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None,15)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        #Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    else :
        #Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix


def color2gray(im_col, lum=True):

     """
    Converts a color image to grayscale using either luminosity or average method.

    Parameters
    ----------
    im_col : 3D array
        Color image array with shape (height, width, channels), typically with 3 channels (R, G, B).
    lum : Boolean, optional
        If True (default), uses the luminosity method to weigh channels according to human perception.
        If False, calculates the average of the RGB channels.

    Returns
    -------
    im_gray : 2D array
        Grayscale image array with shape (height, width), representing the converted grayscale image.
    """

    #Luminosity method
    if lum == True:
        im_gray = im_col[:,:,0]*0.3 + im_col[:,:,1]*0.59 + im_col[:,:,2]*0.11

    #Average method
    if lum == False: 
        im_gray = im_col[:,:,0]/3. + im_col[:,:,1]/3. + im_col[:,:,2]/3.

    return im_gray


def data_prep_nuc_cell_match(file_mask_nuc, file_mask_cell, graphics=False):
    
    """
    Matches nuclei and cell masks and performs cell level QC, optionally displaying segmentation graphics.

    Parameters
    ----------
    file_mask_nuc : str
        Filepath to the nuclei mask file.
    file_mask_cell : str
        Filepath to the cell mask file.
    graphics : bool, optional
        If True, prints graphics of the segmentation process for visualization. Defaults to False.

    Returns
    -------
    mask_nuc : array
        Mask of nuclei after processing and filtering for analysis.
    mask_cell : array
        Mask of cells after processing and filtering for analysis.
    relabel_nucs : array
        Nuclei mask relabeled to match cell labels for direct comparison and easier plotting.
    drop_count : int
        Number of objects (nuclei and cells) rejected from analysis due to various criteria such as mismatch or area constraints.
    """
    
    # Read masks, handling both file paths and pre-loaded arrays
    if type(file_mask_nuc) == str:
        mask_nuc_raw = skimage.io.imread(file_mask_nuc,plugin='pil')
        mask_cell_raw = skimage.io.imread(file_mask_cell,plugin='pil')

    else:
        mask_nuc_raw = file_mask_nuc
        mask_cell_raw = file_mask_cell
    
    #Matching cells to nuclei based on overlap and calculating areas
    matching = {}
    for n in np.unique(mask_nuc_raw.flatten()):
        n = int(n)
        matching[n] = {}
        matching[n]['cyto_match'] = np.argmax(np.bincount(list(mask_cell_raw[mask_nuc_raw==n]))) # find most frequent cyto_mask number within nuc_mask
        matching[n]['nuc_area'] = sum(sum(np.ma.masked_where(mask_nuc_raw==n,mask_nuc_raw).mask)) # nucleus pixel area
        matching[n]['cyto_area'] = sum(sum(np.ma.masked_where(mask_cell_raw==matching[n]['cyto_match'],mask_cell_raw).mask)) # cytoplasm pixel area   
        matching[n]['ncr'] = matching[n]['nuc_area']/matching[n]['cyto_area'] # nucleus - cytoplasm ratio

    df_match = pd.DataFrame(matching).T

    #Remove nuclei without cells, cells with 2 nuclei, cells whose NCR>1 and cells with no nuclei
    droplist_n = list(df_match[df_match.ncr>1.00].index) + list(df_match[df_match.cyto_match==0].index) + list(df_match[df_match.cyto_match.duplicated(keep=False)].index) 
    droplist_c = list(set(mask_cell_raw.flatten())-set(df_match.cyto_match))

    mask_nuc = mask_nuc_raw.copy()
    mask_cell = mask_cell_raw.copy()
    
    #Handle dropping
    for c in droplist_n:
        mask_nuc[np.ma.masked_where(mask_nuc==c,mask_nuc).mask] = 0 # make dropped nuclei into background
        mask_cell[np.ma.masked_where(mask_cell==df_match.cyto_match[c],mask_cell).mask] = 0 # make dropped cell into background
    for c in droplist_c:
        mask_cell[np.ma.masked_where(mask_cell==c,mask_cell).mask] = 0 # make dropped cell into background
        
    cells = df_match.drop(droplist_n)
    cells['nuc_match'] = cells.index
    
    #Create the relabeled nuclear mask
    relabel_nucs = np.zeros((mask_nuc.shape))
    for nuc in cells['nuc_match']:
        relabel_nucs[np.where(mask_nuc == int(nuc))] = int(cells.loc[cells['nuc_match'] == nuc, 'cyto_match'].to_list()[0])

    #Graphics
    if graphics==True:
        plt.imshow(skimage.color.label2rgb(mask_nuc_raw,bg_label=0))
        plt.imshow(skimage.color.label2rgb(mask_cell_raw,bg_label=0))
        plt.imshow(skimage.color.label2rgb(mask_nuc,bg_label=0))
        plt.imshow(skimage.color.label2rgb(mask_cell,bg_label=0))
    
    return mask_nuc, mask_cell, relabel_nucs, len(droplist_n+droplist_c)


def mask_heatmap(mask,props,identity): #Old function only included here for completeness, use updated one in Pipeline_part3

    """
    Maps properties from a DataFrame to a labeled mask, assigning each unique label in the mask
    a value based on a specified property.

    Parameters
    ----------
    mask : array
        A 2D numpy array representing a labeled cell mask, where background is assumed to be 0 and each
        unique label corresponds to a different cell (or more general: region).
    props : DataFrame
        A pandas DataFrame containing properties for each cell. Must include a 'label'
        column that matches labels in the mask and a column for the specified identity to map.
    identity : str
        The column name in props DataFrame whose values are to be mapped to the corresponding labels
        in the mask.

    Returns
    -------
    int_mask : array
        A numpy array of the same shape as the cell mask, where each label in the original mask is replaced
        with the corresponding value from the 'identity' column in props.
    """

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
    
     """
    Calculates skewness of the staining distribution within cells based on the normalized distance between weighted and actual centroids.

    Parameters
    ----------
    weighted_centroids_x : array
        1D array containing the x-coordinates of the staining intensity weighted centroids for each cell.
    weighted_centroids_y : array
        1D array containing the y-coordinates of the staining intensity weighted centroids for each cell.
    centroids_x : array
        1D array containing the x-coordinates of the non-weighted centroids for each cell.
    centroids_y : array
        1D array containing the y-coordinates of the non-weighted centroids for each cell.
    equivalent_diameter : array
        1D array containing the diameter of a circle with equivalent area for each cell.

    Returns
    -------
    out : ndarray
        A 1D numpy array containing the skewness_of_stain metric for each cell, representing the normalized skewness of staining distribution within the cell.

    Notes
    -----
    The equivalent diameter used for normalization is divided by 2 to calculate the normalization factor, as the metric aims to express the distance in terms of radii rather than diameter.
    """
    
    #Load data into arrays
    weighted_centroids = np.vstack((weighted_centroids_x, weighted_centroids_y)).T
    centroids = np.vstack((centroids_x, centroids_y)).T
    
    out = np.array([])

    for cell in range(0, equivalent_diameter.size, 1):
        
        #Calculate Euclidean distance between actual and weighted centroids
        dist = np.linalg.norm((weighted_centroids[cell, :] - centroids[cell, :]))

        #Normalize by cell radius
        norm_dist = dist / (equivalent_diameter[cell] / 2)
        out = np.append(out, norm_dist)
    
    return out
        

def centrality_of_stain(images, intensity_images):
    
    """
    Calculates the "centrality of stain" metrics within cells, providing insights into the spatial distribution of staining intensity from cell center to periphery.

    Parameters
    ----------
    images : dict
        A dictionary containing binary images (masks) for each cell object. Each key represents a cell identifier with its corresponding binary image as the value.
    intensity_images : dict
        A dictionary containing the actual intensity images for each cell object. Each key should correspond to the same key in `images` dict, with its value being the intensity image of the cell.

    Returns
    -------
    out_ratio : ndarray
        A 1D numpy array containing the mean weighted ratio of intensity means for each cell, reflecting the gradient of staining intensity from the edge towards the center.
    out_tendency : ndarray
        A 1D numpy array containing the calculated concentration of staining intensity towards the center of each cell.
    out_spearman : ndarray
        A 1D numpy array containing the Spearman correlation coefficients for each cell, reflecting the relationship between erosion steps and staining intensity.

    Notes
    -----
    - Not used in further analysis.
    """
    
    out_ratio = np.array([])
    out_tendency = np.array([])
    out_spearman = np.array([])
    
    for cell in range(0, images.size, 1):
        
        mask = images[cell].astype(int)
        image = intensity_images[cell]
        means = np.array([])

        #Iteratively erode the mask and calculate mean intensities
        while True:
            erode_1 = scipy.ndimage.morphology.binary_erosion(mask, iterations = 1)
            measure = mask - erode_1
            temp_mean = np.mean(image[measure.astype(np.bool)])
            if np.isnan(temp_mean):
                break
            means = np.append(means, temp_mean)
            mask = erode_1.astype(int)
        
        #Perform metric calculations
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
    
    """
    Calculates the proportion of pixels that cumulatively add up to a specified percentage of the total staining intensity within cells.

    Parameters
    ----------
    images : array or list
        A collection of binary masks for each cell, where each mask defines the boundary of the cell.
    intensity_images : array or list
        A collection of intensity images for each cell, corresponding to the actual staining intensity observed within the cell boundaries.
    prop : float
        The proportion of the total staining intensity to consider. For example, a value of 0.5 corresponds to 50% of the total staining intensity.

    Returns
    -------
    out : ndarray
        A 1D numpy array where each element represents the proportion of pixels of the total number of stained pixels in a cell that cumulatively add up to the specified proportion of the cell's total staining intensity.

    Notes
    -----
    - This metric is useful for assessing the heterogeneity of staining within cells, identifying cells with localized areas of high intensity versus those with more uniform staining distribution.
    """
    
    out = np.array([])
    
    for cell in range(0, images.size, 1):

        mask = images[cell].astype(np.bool)
        image = intensity_images[cell]

        #Extract and sort pixel intensities within the mask in descending order
        measure = image[mask].flatten()
        measure = np.sort(measure, kind='stable')[::-1]

        #Calculate the cumulative sum required to reach the specified proportion of total intensity
        arr_sum = np.sum(measure) * prop
        
        count_sum = 0
        counter = 0
        for element in measure:
            if count_sum < arr_sum:
                counter += 1
                count_sum += element
            else:
                break

        #Calculate and save the proportion of pixels needed
        prop_needed = counter / measure.size
        out = np.append(out, prop_needed)
        
    return out


def mutual_information_2d(x, y, normalized=True, nbins: int):
    
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
    nbins: Integer
        Number of bins to use for histogram calculation

    Returns
    -------
    mi : float
        Mutual information as a similarity measure
    """

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

    """
    Computes the entropy of Mutual Information (MI) scores across sub-tiles of two images, providing a measure of the overall similarity and complexity of the relationship between the two images.

    Parameters
    ----------
    im1 : ndarray
        The first image array. It should be a 2D numpy array.
    im2 : ndarray
        The second image array, of the same dimensions as `im1`.

    Returns
    -------
    entr : float
        The entropy of the Mutual Information scores calculated from the sub-tiles of `im1` and `im2`.

    Notes
    -----
    - Not used in further analysis.
    - The images are divided into 100x100 pixel sub-tiles, but this can be adjusted based on the size of the input images and the desired resolution for analysis.
    - Normalized Mutual Information is calculated and adjusted by squaring the inverse of the MI value for each sub-tile, to emphasize differences.
    """
    
    #Set up tiling
    x, y = im1.shape
    num_tilesx = im1.shape[0]//100
    num_tilesy = im1.shape[1]//100
    step_x = int((x - (x % num_tilesx)) / num_tilesx)
    step_y = int((y - (y % num_tilesy)) / num_tilesy)
    
    MI_results = []
    
    for i in range(1, (num_tilesx + 1)):
        for z in range(1, (num_tilesy + 1)):

            #Extract tiles
            temp1 = im1[((i * step_x) - step_x) : (i * step_x), ((z * step_y) - step_y) : (z * step_y)]
            temp2 = im2[((i * step_x) - step_x) : (i * step_x), ((z * step_y) - step_y) : (z * step_y)]

            MI_results.append((1/mutual_information_2d(temp1.ravel(), temp2.ravel(), normalized=True, nbins=round(np.sqrt((temp1.size/5)))))**2)
     
    #Calculate entropy       
    entr = scipy.stats.entropy(MI_results, base=len(MI_results))
    
    return entr


def optimizing_func(x, raw, sub):
    
    """
    For usage in optimization by minimizing the Mutual Information (MI) between a subtracted image and a reference image. 
    This function calculates the MI between the reference image and a modified version of the subtracted image.

    Parameters
    ----------
    x : float
        The scaling factor applied to the subtracted image before performing the subtraction from the raw image.
    raw : ndarray
        The raw reference image array. It should be a 2D numpy array representing the image from which we aim to subtract.
    sub : ndarray
        The subtracted image array, which will be scaled by `x` and subtracted from `raw`. It should be of the same dimensions as `raw`.

    Returns
    -------
    float
        The Mutual Information (MI) score between the modified subtracted image (opt_im) and the reference image.
    """

    opt_im = raw - x*sub

    return (mutual_information_2d(sub.ravel(), opt_im.ravel(), normalized=True, nbins=round(np.sqrt((opt_im.size/5)))))


def Minimize_MI(imDict):
    
    """
    Minimizes Mutual Information (MI) between quench subtracted (qsub) images across imaging cycles.

    Parameters
    ----------
    imDict : dict
        A dictionary with keys in the format "cycleX", where X is the cycle number (0 to 7), 
        and values are image matrices representing the images at each cycle.

    Returns
    -------
    outDict : dict
        A dictionary where each key is a cycle (after the first) and each value is a sub-dictionary. 
        The sub-dictionary keys are previous cycles, and values are the scaling factors (alpha) that minimized the MI 
        between the current cycle image and the quench subtracted images from previous cycles.

    Notes
    -----
    - This function optimizes images by iteratively minimizing MI with previous cycles 
      by finding optimal scaling factors between a current cycle image and its preceding cycles' images.
    - The optimization uses a polynomial fit of degree 5 to the optimizing_func results to find the minimum MI scaling factor for subtraction.
    - The final corrected image for each cycle is constrained to have non-negative values.
    """

    cycle_list = sorted(imDict.keys())
    
    corrDict = {}
    outDict = {}
    corrDict[cycle_list[0]] = imDict[cycle_list[0]]
    
    for i in range(1, len(cycle_list)):

        cycle = cycle_list[i]
        Im_raw = imDict[cycle]
        Im_use = cycle_list[:i]
        subDict = {}
        
        for cyc in Im_use:

            Im = corrDict[cyc]
            x_ran = np.arange(0,1,0.02)
            y_ran = np.array([])

            #Evaluate the range of scaling factors (x_ran) with optimizing_func
            for x in x_ran:
                y_ran = np.append(y_ran, optimizing_func(x, raw=Im_raw, sub=Im))
            
            #Fit polynomial of degree 5 (good balance between stability and goodness of fit)
            polyfit = np.polynomial.polynomial.Polynomial.fit(x=x_ran, y=y_ran, deg=5)
            fit = polyfit.linspace(n=10000)

            #Extract scaling factor that minimized MI
            x_min = fit[0][np.argmin(fit[1])]
            subDict[cyc] = x_min
 
        for prev, alpha in subDict.items():
            Im_raw = Im_raw - alpha*corrDict[prev]
            
        corrDict[cycle] = np.maximum(0, Im_raw)  
        outDict[cycle] = subDict
    
    return outDict


def Correct_im(imDict, alphas):
    
    """
    Applies corrections to images based on predetermined alpha values calculated with Minimize_MI and median_alpha to minimize Mutual Information (MI) with previous cycles.

    Parameters
    ----------
    imDict : dict
        A dictionary where keys are in the format "cycleX" (X being the cycle number) and values are image matrices for each cycle.
    alphas : dict
        A dictionary containing the alpha values for correction. Each key corresponds to a cycle, and its value is another dictionary 
        mapping previous cycles to their respective alpha values used to minimize MI

    Returns
    -------
    corrDict : dict
        A dictionary with corrected image matrices for each cycle. The corrections are applied by subtracting the scaled images of previous cycles,
        based on the alpha values that minimize the MI between the current and previous cycle images. The corrected images are constrained to non-negative values.
    """

    cycle_list = sorted(imDict.keys())
    
    corrDict = {}
    corrDict[cycle_list[0]] = imDict[cycle_list[0]]
    
    for i in range(1, len(cycle_list)):

        cycle = cycle_list[i]
        Im_raw = imDict[cycle]
        Im_use = cycle_list[:i]
        subDict = alphas[cycle]
        
        #Iteratively minimize MI with previous cycles
        for prev, alpha in subDict.items():
            Im_raw = Im_raw - alpha*corrDict[prev]
            
        corrDict[cycle] = np.maximum(0, Im_raw)
    
    return corrDict


def annotate_artifacts(int_mask, im1, im2, int_thres_global, int_thres_local, ratio_thres, rad):
    
    """
    Annotates artifacts in images by comparing global and local frequency amplitude ratios and intensity differences.

    Parameters
    ----------
    int_mask : ndarray
        cell mask
    im1, im2 : ndarray
        Aligned and cropped images to be assessed, not quench subtracted
    int_thres_global : int or float
        Global intensity threshold to identify significant differences between images.
    int_thres_local : int or float
        Local intensity threshold for assessing localized differences between images.
    ratio_thres : float
        Threshold for frequency amplitude ratios used in artifact detection.
    rad : int
        Radius used for frequency filtering to distinguish between low and high frequency components.

    Returns
    -------
    list
        A list of strings indicating whether each cell passes or fails artifact annotation based on global and local frequency ratios.

    Notes
    -----
    - The function first assesses global artifacts by comparing overall intensity and frequency ratios.
    - If global criteria are not met, it proceeds to further analyze possible artifacts locally.
    - Artifacts are identified based on deviations in intensity and frequency amplitude ratios beyond specified thresholds.
    """
    
    #Get list of cell identifiers
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
    
    #Global thresholding
    if FratioG < ratio_thres or (np.mean(img2) - np.mean(img1)) < int_thres_global:

        outstring = '_'.join(['PassG', str(FratioG), '0'])

        return [outstring]*(mask_unique.size)
    
    else:

        #Recreate the images only using the low frequency spectrum

        #Initialize complex array
        artifact_mask = np.zeros(measure.shape, dtype='complex128')

        #Enter quench-subtracted (in fourier space) low frequency values
        artifact_mask[rr,cc] = FFT2[measure == 1] - FFT1[measure == 1]

        #Inverse FFT
        recreate_img = np.abs(scipy.fft.ifft2(scipy.fft.fftshift(artifact_mask)))
        
        #Threshold and label
        thres = threshold_mean(recreate_img)
        recreate_img = recreate_img > thres
        labels = skimage.measure.label(recreate_img, connectivity=2)   

        #Check affected areas

        #Initalize dictionary to contain the FRatio values for all affected areas
        check_dict = {0:0}

        for c in np.unique(labels)[1:]:

            obj_coord = np.where(labels==c)

            #Min. Size threshold of 2500 pixels
            if obj_coord[0].size > 2500: 
                
                #Measure local FRatio (same approach as global)
                obj1 = img1[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1]
                obj2 = img2[np.min(obj_coord[0]):np.max(obj_coord[0])+1, np.min(obj_coord[1]):np.max(obj_coord[1])+1]
                
                if (np.mean(obj2) - np.mean(obj1)) >= int_thres_local:

                    measure = np.zeros(obj1.shape)
                    rr, cc = draw.disk((measure.shape[0]/2, measure.shape[1]/2), radius=rad, shape=measure.shape)
                    measure[rr,cc] = 1

                    FFT1 = scipy.fft.fftshift(scipy.fft.fft2(obj1))
                    FFT2 = scipy.fft.fftshift(scipy.fft.fft2(obj2))
                    FFTsub = np.abs(FFT2) - np.abs(FFT1)
                    FratioL = np.mean(FFTsub[measure == 1]) / np.maximum(1, np.mean(FFTsub[measure == 0]))

                    check_dict[c] = FratioL
        
        #Create and return output list
        out_list = []
            
        for cell in mask_unique:

            regions = np.unique(labels[int_mask == cell])
            rel_keys = sorted(list(check_dict.keys()))[1:]
            cell_ratios = [check_dict[region] for region in regions if region in rel_keys]
            cell_ratios.append(1)
            max_ratio = max(cell_ratios)
            
            if max_ratio < ratio_thres:
                out_list.append('_'.join(['PassL', str(FratioG), str(max_ratio)]))

            else:
                out_list.append('_'.join(['FailL', str(FratioG), str(max_ratio)]))
        
        return out_list


def median_alpha(inDict): 
    
    """
    Computes the median alpha values across all provided FOVs for all cycles and channels to be used for final corrections.

    Parameters
    ----------
    inDict : dict
        A dictionary where each key is an FOV identifier with a nested dictionary as its value. The nested dictionaries
        are those containing the alpha values that minimize MI for a given FOV created by Minimize_MI.

    Returns
    -------
    outDict : dict
        A dictionary containing median MI minimizing alpha values (median across all provided FOVs) for every pair of cycle and previous cycle.
    """
    
    outDict = inDict[list(inDict.keys())[0]]
    
    for cycle, subs in outDict.items():
        for cyc, alpha in subs.items():

            collect_array = np.array([])

            for im in inDict.keys():
                collect_array = np.append(collect_array, inDict[im][cycle][cyc])

            outDict[cycle][cyc] = np.median(collect_array)
            
    return outDict

       
#Extra function for scikit-image regions_props that calculates median fluorescence intensity
def p50(regionmask,intensity_image):
    return np.quantile(intensity_image[regionmask],0.50)