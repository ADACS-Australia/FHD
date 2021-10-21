import numpy as np
from fhd_utils.weight_invert import weight_invert
from fhd_core.gridding.visibility_count import visibility_count
from fhd_core.setup_metadata.fhd_save_io import fhd_save_io

def filter_uv_uniform(image_uv, obs, psf, params, weights, fi_use = None, bi_use = None, 
                      mask_mirror_indices = None, name = "uniform", return_name_only = False):
    """
    TODO: Description

    image_uv: array
        The image we are applying a filter to
    obs: dict
        Data Structure full of values required for calculation
    psf: dict
        Data Structure full of values required for calculation
    params: dict
        Data Structure with values which are lists
    weights: array
        Array of weights for the filter
    filter: array
        The kernel for the filter, same size as image_uv, if not then it goes through
        to a filter function.
    name: String, optional
        By default its set to "uniform"
    return_name_only: bool, optional
        By default is False, set to True to only return the name and image_uv as it was inputted.
    
    Returns
    -------
    image_uv: array
        A filtered image_uv array
    """
    # Return the name now, if that has been set
    if return_name_only:
        # Why do you want this?
        return image_uv, name
    # This does not make use of fine-grained flagging, but relies on coarse flags from the obs structure 
    # (i.e. a list of tiles completely flagged, and of frequencies completely flagged)
    if obs is None or psf is None or params is None:
        raise TypeError("obs, psf and params must not be None")
    vis_count = visibility_count(obs, psf, params, weights, fi_use, bi_use, mask_mirror_indices)
    # Get the parts of the filter we're using
    filter_use = weight_invert(vis_count, threshold = 1)
    # Get the weights index as well
    if np.size(weights) == np.size(image_uv):
        wts_i = np.where(weights)
    else:
        wts_i = np.where(filter_use)
    # Apply a mean normalization
    if np.size(wts_i) > 0:
        filter_use /= np.mean(filter_use[wts_i])
    else:
        filter_use /= np.mean(filter_use)
    #Return the filtered
    return image_uv * filter_use, name               


