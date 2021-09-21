import numpy as np
from fhd_utils.modified_astro.meshgrid import meshgrid
from fhd_utils.histogram import histogram

def holo_mapfn_convert(map_fn, psf_dim, dimension, elements = None, norm = 1, threshold = 0):
    """
    TODO: Description

    Parameters
    ----------
    map_fn: ndarray
        TODO: Description
    psf_dim: int, float
        TODO: Description
    dimension: int, float
        TODO: Description
    elements: None, optional
        TODO: Description
    norm: int
        TODO: Description
    threshold: int, float, optional
        TODO: Description

    Returns
    -------
    
    """
    # Set up all the necessary arrays and numbers
    if elements is None:
        elements = dimension
    psf_dim2 = 2 * psf_dim
    psf_n = psf_dim ** 2
    psf_i = np.arange(psf_n)
    sub_xv = meshgrid(psf_dim2, 1) - psf_dim
    sub_yv = meshgrid(psf_dim2, 2) - psf_dim
    n = dimension * elements
    # Generate an array of shape elements x dimension
    n_arr = np.zeros((elements, dimension))
    # Get the amount of elements that meet the threshold...although its all zeros...?
    n1 = np.size(np.where(n_arr[1: elements - 1, 1 : dimension - 1] > threshold))
    # Set the size...? This is weird?
    n_arr[1: elements - 1, 1 : dimension - 1] = n1
    # Get the ones to use
    i_use = np.where(n_arr)
    # Get the amount we're using
    i_use_size = np.size(i_use)
    # If we aren't using any then return 0
    if i_use_size == 0:
        return 0
    
    # Get the reverse indices
    _, _, ri = histogram(i_use, min = 0)
    # Create zeros of the same size as what we're using
    sa = np.zeros(i_use_size)
    ija = np.zeros(i_use_size)
    # Fill in the sa and ija arrays
    for index in range(1, i_use_size):
        i = i_use[index]
        xi = np.floor(i / dimension)
        yi = i % dimension
        map_fn_sub = map_fn[xi, yi]
        j_use = np.where(np.abs(map_fn_sub) > threshold)

        xii_arr = sub_xv[j_use] + xi
        yii_arr = sub_yv[j_use] + yi
        sa[index] = map_fn_sub[j_use]
        ija[index] = ri[ri[xii_arr * dimension+ yii_arr]]
    # Return a dictionary
    return {"ija" : ija, "sa" : sa, "i_use": i_use, "norm": norm, "indexed" : 1}





            