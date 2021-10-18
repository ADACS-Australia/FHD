from baseline_grid_locations import baseline_grid_locations
from conjugate_mirror import conjugate_mirror
import numpy as np

def visibility_count(obs, psf, params, vis_weight_ptr, xmin, ymin, fi_use, bi_use, mask_mirror_indices,
                     file_path_fhd = None, no_conjugate = True, fill_model_visibilities = False):
    """[summary]

    Parameters
    ----------
    obs : [type]
        [description]
    psf : [type]
        [description]
    params : [type]
        [description]
    vis_weight_ptr : [type]
        [description]
    xmin : [type]
        [description]
    ymin : [type]
        [description]
    fi_use : [type]
        [description]
    n_freq_use : [type]
        [description]
    bi_use : [type]
        [description]
    mask_miiror_indices : [type]
        [description]
    file_path_fhd : [type], optional
        [description], by default None
    no_conjugate : bool, optional
        [description], by default True
    fill_model_vis : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    TypeError
        [description]
    """
    
    if obs is None or psf is None or params is None or vis_weight_ptr is None:
        raise TypeError("obs, psf, params or vis_weight_ptr should not be None")
    
    #Retrieve info from the data structures
    dimension = obs['dimension']
    elements = obs['elements']
    psf_dim = psf['dim']

    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weight_ptr, bi_use = bi_use, fi_use = fi_use,
                                             mask_mirror_indices = mask_mirror_indices, fill_model_visibilities = fill_model_visibilities)
    # Retrieve the data we need from baselines_dict
    bin_n = baselines_dict['bin_n']
    bin_i = baselines_dict['bin_i']
    xmin = baselines_dict['xmin']
    ymin = baselines_dict['ymin']
    ri = baselines_dict['ri']
    n_bin_use = baselines_dict['n_bin_use']
    # Remove baselines_dict
    del(baselines_dict)

    uniform_filter = np.zeros((elements, dimension))
    for bi in range(n_bin_use):
        inds = ri[ri[bin_i[bi]] : ri[bin_i[bi] + 1] - 1]
        ind0 = inds[0]
        xmin_use = xmin[ind0]
        ymin_use = ymin[ind0]
        uniform_filter[ymin_use : ymin_use + psf_dim - 1, xmin_use : xmin_use + psf_dim - 1] += bin_n[bin_i[bi]]
    
    if not no_conjugate:
        uniform_filter = (uniform_filter + conjugate_mirror(uniform_filter)) / 2

    # TODO: Write uniform_filter to file? 
    # fhd_save_io
    
    return uniform_filter



    