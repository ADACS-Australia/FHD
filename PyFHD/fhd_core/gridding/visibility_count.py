from _typeshed import NoneType
from fhd_core.setup_metadata.fhd_save_io import fhd_save_io
from fhd_utils.array_match import array_match
from fhd_utils.histogram import histogram
import numpy as np

def visibility_count(obs, psf, params, vis_weight_ptr, file_path_fhd = None,
                     no_conjugate = False, fill_model_vis = False, *args, **kwargs):
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
    file_path_fhd : [type], optional
        [description], by default None
    no_conjugate : bool, optional
        [description], by default False
    fill_model_vis : bool, optional
        [description], by default False
    
    Returns
    -------
    weights: [type]
        [description]
    
    Raises
    ------
    NoneType
        Raised when obs, psf and params is None and no file_path_fhd is given
        Although fhd_save_io is currently incomplete, so a NoneType error will
        occur at the moment even if file_path_fhd is set.
    """
    if file_path_fhd is None and(obs is None or psf is None or params is None):
        raise NoneType("obs, psf or params is None while no file_path_fhd is supplied to restore it")
    elif obs is None and file_path_fhd is not None:
        # How fhd_save_io works is up for discussion at the moment
        # As such if any of the obs, psf and params is None a NoneType error will get thrown
        # TODO: fhd_save_io()
        fhd_save_io() 
    elif psf is None and file_path_fhd is not None:
        fhd_save_io()
    elif params is None and file_path_fhd is not None:
        fhd_save_io()
    
    n_pol = obs['n_pol']
    n_tile = obs['n_tile']
    n_freq = obs['n_freq']
    dimension = obs['dimension']
    elements = obs['elements']
    kbinsize = obs['kpix']
    ky_span = kx_span = kbinsize * dimension
    min_baseline = obs['min_baseline']
    max_baseline = obs['max_baseline']
    b_info = obs['baseline_info']

    freq_bin_i = b_info['fbin_i']
    fi_use = np.where(b_info['freq_use'])
    if fill_model_vis:
        fi_use = np.arange(n_freq)
    freq_bin_i = freq_bin_i[fi_use]

    frequency_array = b_info['freq']
    frequency_array = frequency_array[fi_use]

    tile_use = np.where(b_info['tile_use']) + 1
    if fill_model_vis:
        tile_use = np.arange(n_tile)  + 1
    bi_use = array_match(b_info['tile_A'], tile_use, array2 = b_info['tile_B'])
    n_b_use = np.size(bi_use)
    n_f_use = np.size(fi_use)

    psf_dim = psf['dim']
    psf_resolution = psf['resolution']

    kx_arr = params['uu'][bi_use] / kbinsize
    ky_arr = params['vv'][bi_use] / kbinsize

    dist_test = np.sqrt((kx_arr **2 + ky_arr **2) * kbinsize)
    dist_test = dist_test * frequency_array
    flag_dist_i = np.where((dist_test < min_baseline) | (dist_test > max_baseline))
    dist_test = 0

    xcen = kx_arr * frequency_array
    ycen = ky_arr * frequency_array

    conj_i = np.where(ky_arr > 0)
    if np.size(conj_i) > 0:
        xcen[conj_i, :] = -xcen[conj_i, :]
        ycen[conj_i, :] = -ycen[conj_i, :]
    
    xmin = (np.floor(xcen) + dimension / 2 - (psf_dim / 2 - 1)).astype("int")
    ymin = (np.floor(ycen) + elements / 2 - (psf_dim / 2 - 1)).astype("int")

    if np.size(flag_dist_i) > 0:
        xmin[flag_dist_i] = -1
        ymin[flag_dist_i] = -1
        flag_dist_i = 0
    
    range_test_x_i = np.where((xmin <= 0) | ((xmin + psf_dim - 1) >= dimension - 1))
    if np.size(n_test_x) > 0:
        xmin[range_test_x_i] = ymin[range_test_x_i] = -1
    range_test_x_i = 0
    range_test_y_i = np.where((ymin <= 0) | ((ymin + psf_dim - 1) >= elements - 1))
    if np.size(range_test_y_i) > 0:
        xmin[range_test_y_i] = ymin[range_test_y_i] = -1
    range_test_y_i = 0

    if vis_weight_ptr is not None and not fill_model_vis:
        flag_i = np.where(vis_weight_ptr[0] <= 0)
        xmin[flag_i] = -1
        ymin[flag_i] = -1
    
    # match all visibilities that map from and to exactly the same pixels
    # should miss any (xmin,ymin)=(-1,-1) from weights
    bin_n, _, ri = histogram(xmin + ymin * dimension, min = 0)
    bin_i = np.where(bin_n)
    weights = np.zeros((elements, dimension))
    for bi in range(np.size(bin_i)):
        inds = ri[ri[bin_i[bi] : ri[bin_i[bi]] + 1] - 1]
        ind0 = inds[0]
        # should all be the same, but don't want an array
        xmin_use = xmin[ind0]
        ymin_use = ymin[ind0]
        weights[ymin_use : ymin_use + psf_dim - 1, xmin_use : xmin_use + psf_dim - 1] += bin_n[bin_i[bi]]
    
    if not no_conjugate:
        weights_mirror = np.roll(np.flipud(np.fliplr(weights)), 1, axis = (0,1))
        weights = (weights + weights_mirror) / 2
    
    # fhd_save_io
    return weights



    