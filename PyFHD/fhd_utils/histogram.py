import numpy as np

def histogram(data, bin_size = 1, num_bins = None, min = None, max = None) :
    """
    This histogram function will replicate the IDL Histogram function
    Its different from NumPy as it separates the number of bins and bin size
    with different behaviour for both. It also returns the indices of the histogram. 
    this function is more of a wrapper function for np.histogram and
    np.histogram_bin_edges. However, its behaviour is not the same as np.histogram.

    Parameters
    ----------
    data: ndarray
        The data used to compute the histogram
    bin_size: float, optional
        Set the bin size (or bin width), the default is 1
    num_bins : int, optional
        Set the number of bins, note this will override bin_size
        if it has been used.
    return_indices: bool
        True by default is to return the indices of the histogram,
        these will just the result of calling numpy.digitize
    min: float, optional
        Set a minimum value the histogram should look from
    max: float, optional
        Set a maximum value the histogram should look at

    Returns
    -------
    hist: ndarray, dtype is int
        The values/counts of the histogram
    bin_edges: ndarray, dtype is float
        The edges of each bin
    indices: ndarray, dtype is int
        An ndarray of the same shape as data
        containing the indexes of each value from bin edges

    Examples
    --------
    >>> data= np.array([[ -5,   4,   2,  -8,   1],
                        [  3,   0,   5,  -5,   1],
                        [  6,  -7,   4,  -4,  -8],
                        [ -1,  -5, -14,   2,   1]])
    >>> hist, bins, indices = histogram(data)
    >>> print(hist)
    [1, 0, 0, 0, 0, 0, 2, 1, 0, 3, 1, 0, 0, 1, 1, 3, 2, 1, 2, 1, 1]
    >>> print(indices)
    [22 23 23 23 23 23 23 25 26 26 29 30 30 30 31 32 35 37 38 40 41 42 17  3
     14 11  0  8 16 13 15  6  4  9 19  2 18  5  1 12  7 10]

    Raises
    ------
    ValueError
        Get's raised if any of the values are not compatible with each other.
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
        bins = num_bins
    # Set the bins using bin_size
    else:
        # IDL uses the bin_size as equal throughout min to max
        bins = np.arange(min , max + bin_size, bin_size)
        # However, if we set a max, we must adjust the last bin to max
        if bins[-1] > max:
            bins[-1] = max
        # In the case the max mod bin_size is 0, add max again, 
        # as IDL treats last bin as [x, y), i.e. right edges all the way to end
        # unlike NumPy which does [x, y] i.e. at end includes all values
        if max % bin_size == 0:
            bins = np.append(bins, max)
    hist, bin_edges = np.histogram(data, bins = bins, range = (min, max))
    # As we purposely added a bin for max to get the same behaviour as IDL remove it now 
    bin_edges = bin_edges[:-1]

    '''
    Get the reverse indices as IDL does it, the result should be
    an array which has two vectors, it should be of size len(hist) + np.size(data) + 1.
    The first vector of size len(hist) contains the indexes for later in the
    array (i.e. the array references itself). The second vector of np.size(data)
    contains all the indexes where the bins occurred in the data. 

    This method can likely be optimized through:
        - Reducing memory use by using the one array which also reduces calls to append
          increase the performance of the function. It will also remove the need to concatenate.
          Which with large arrays will likely be a huge cost
        - At some point SciPy's Sparse Matrices may be useful depending on the data
        - If we do replace this with np.digitize at some point, we either need a fast way
          to use the results of digitize to apply the results to a particular bin (which is what
          REVERSE_INDICES does very well) or a fast way to convert np.digitize to the same format
          as REVERSE_INDICES.
        - Basically we just need a way to vectorize the code below
    '''
    # Initialise our two vectors
    # The first keeps track of the indexes in the second vector
    first_vector = np.array([], dtype = int)
    # This keeps track of all the occurrences of the bins in the data (when flattened)
    second_vector = np.array([], dtype = int)
    # We used len(bin_edges) a lot, let's just store it
    bin_len = len(bin_edges)
    #Get the current index
    index  = bin_len + 1
    # Add the first index to the first vector
    first_vector = np.append(first_vector, np.array([index]))
    # Loops through bin_edges array
    for bin in range(bin_len):
        # Get an array of all the occurrences
        if bin == bin_len - 1:
            indexes_for_bin = np.where((data.flatten() >= bin_edges[bin]) & (data.flatten() <= max))[0]
        else:
            indexes_for_bin = np.where((data.flatten() >= bin_edges[bin]) & (data.flatten() < bin_edges[bin + 1]))[0]
        # Add the found bin values indexes to the second vector 
        second_vector = np.append(second_vector, indexes_for_bin)
        # We now need to update where to find the second vector index values
        index += len(indexes_for_bin)
        # Add the updated value to the first index
        first_vector = np.append(first_vector, np.array([index]))
    # Concatenate the first and second vector
    # The first np.size(hist) values now reference indexes within the same array
    # data[indices[indices[I] : indices[i+1]-1]] is used to access the actual bin value from the data array
    indices = np.concatenate([first_vector, second_vector])

    #Return histogram, bin_edges and the indices
    return hist, bin_edges, indices