from histogram import histogram
import numpy as np

def array_match(array_1, value_match, array_2 = None) :
    """
    TODO: Description for array match

    Parameters
    ----------
    array_1: array
        TODO: Add Description for Array_1
    value_match: array
        TODO: Add Description for Value_Match
    array_2: array, optional
        TODO: Add Description for Array_2

    Returns
    -------
    matching indices: array
        TODO: Add Description for return of array_match
    """
    if value_match is not None and value_match.match.shape[0] == 0:
        value_match = [value_match]
    # If array_2 has been supplied, compare which mins and maxes to use based on two arrays
    if array_2 is not None and len(array_2) > 0:
        min_use = np.amin(np.amin(array_1), np.amin(array_2))
        max_use = np.amax(np.amax(array_1), np.amax(array_2))
        # Also compute the histogram for array_2
        hist2, _, ri2 = histogram(array_2, min = min_use, max = max_use)
    else:
        # If the second array wasn't supplied
        min_use = np.amin(array_1)
        max_use = np.amax(array_1)
        # Supply a second hist
        hist2 = np.arange(max_use - min_use + 1)
    # Get the histogram for the first   
    hist1, _ , ri1 = histogram(array_1, min = min_use, max = max_use)
    # Arrays should be the same size, does addition
    hist_combined = hist1 + hist2
    bins = np.where(hist_combined > 0)

    # Select the values to be used
    hist_v1, bins_v1, _ = histogram(bins+min_use)
    omin = bins_v1[0]
    omax = bins_v1[-1]
    hist_v2, _, _ = histogram(value_match, min = omin, max = omax)
    vals = np.where(np.bitwise_and(hist_v1, hist_v2))
    n_match = len(vals)

    if len(vals) == 0:
        return -1
    
    ind_arr = np.zeros_like(array_1)
    for vi in range(n_match - 1):
        i = vals[vi]
        if hist1[i] > 0:
            ind_arr[ri1[ri1[i] : ri1[i+1]]] = 1
        if hist2[i] > 0:
            ind_arr[ri2[ri2[i] : ri2[i+1]]] = 1
    
    # Return our matching indices
    return np.where(ind_arr)
