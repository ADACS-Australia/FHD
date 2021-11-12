import numpy as np
from numba import njit, prange

@njit(parallel = True)
def get_bins(min, max, bin_size):
    """
    TODO: Docstrings for Hist

    Parameters
    ----------
    min : [type]
        [description]
    max : [type]
        [description]
    bin_size : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return np.arange(min , max + bin_size, bin_size)

@njit(parallel = True, fastmath = True)
def get_hist(data, bins, max):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    bins : [type]
        [description]
    max : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Setup the dictionary
    bin_l = bins.size
    indexes = {}
    # Setup the histogram
    hist = np.zeros(bin_l, dtype = np.int64)
    # Find the bins in the data
    for bin_i in prange(bin_l):
        # Get the indexes that where bins[bin_i] is in data
        if bin_i == bin_l - 1:
            indexes[bin_i] = np.where((data >= bins[bin_i]) & (data <= max))[0]
        else:
            indexes[bin_i] = np.where((data >= bins[bin_i]) & (data < bins[bin_i + 1]))[0]
        # The size of what has been found is the size for the histogram
        hist[bin_i] = indexes[bin_i].size
    return hist, indexes

def get_ri(indexes, hist, bin_l):
    """[summary]

    Parameters
    ----------
    indexes : [type]
        [description]
    hist : [type]
        [description]
    bin_l : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # The data indexes start at bin_l + 1
    index = bin_l + 1
    # Set up the reverse indices array
    ri = np.zeros(bin_l + np.sum(hist) + 1, dtype = np.int64)
    # Fill out the reverse indices using the indexes dict which already contains the indexes
    keys = np.sort(list(indexes.keys()))
    for key in keys:
        ri[key] = index
        ri[index : index + indexes[key].size] = indexes[key]
        index += indexes[key].size
    # Set the last index
    ri[bin_l] = index
    return ri

def histogram(data, bin_size = 1, num_bins = None, min = None, max = None):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    bin_size : int, optional
        [description], by default 1
    num_bins : [type], optional
        [description], by default None
    min : [type], optional
        [description], by default None
    max : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    # Do the error checks
    if (min is not None and max is not None and min > max) :
        raise ValueError("Your minimum is higher than your maximum, check your min and max")
    # If the minimum has not been set, set it
    if min is None:
        min = np.min(data)
    # If the maximum has not been set, set it
    if max is None:
        max = np.max(data)
    # If the number of bins has been set use that
    if num_bins is not None:
        bin_size = (max - min) / num_bins
    # IDL uses the bin_size as equal throughout min to max
    bins = get_bins(min, max, bin_size)
    # However, if we set a max, we must adjust the last bin to max according to IDL specifications
    if bins[-1] > max or num_bins is not None:
        bins = bins[:-1]
    # Flatten the data
    data_flat = data.flatten()
    # Use the parallelized numba function to get the histogram and the data indexes
    hist, indexes = get_hist(data_flat, bins, max)
    # Use the numba function not parallized to get the reverse indices
    ri = get_ri(indexes, hist, bins.size)

    # return
    return hist, bins, ri