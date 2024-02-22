'''
Functions for part 1 of main HNSCC image analysis pipeline: Illumination correction
author: Jan Hoelzl

Center for Systems Biology
Massachusetts General Hospital
'''



##FUNCS

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