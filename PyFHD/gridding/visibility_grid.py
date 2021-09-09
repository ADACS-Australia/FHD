import numpy as np
import datetime

def visibility_grid(visibility, vis_weights, obs, status_str, psf, params,
                    file_path_fhd= "/.", weights = False, variance = False, polarization = 0,
                    mapfn_recalculate = False, uniform_filter = False, fi_use = [],
                    bi_use = [], no_conjugate = False, return_mapfn = False, mask_mirror_indices = False,
                    no_save = False, model = [], model_flag = False, preserve_visibilities = False,
                    error = False, grid_uniform = False, grid_spectral = False, spectral_model_uv = 0,
                    beam_per_baseline = False, uv_grid_phase_only = False, *args, **kwargs) :
    """
    TODO: Find Out what it does for the doc
    TODO: Get Defaults!
    ...

    Parameters
    ----------

    visibility: Type
        Pointer in IDL, Description to be added
    vis_weights: array
        An array of visibility weights
    obs: dict
        Contains options which set behaviour in the function
    status_str: str
        A Status String of some sort
    psf: dict
        Contains options which set behaviour in the function
    params: dict
        Is a dictionary whose values are lists

    Returns
    -------
    image_uv : Type
        TODO: Description goes here

    Examples
    --------
    >>> visibility_grid()
    Test1

    >>> visibility_grid()
    Test2

    >>> visibility_grid()
    Test3
    """
    # Get the current time
    t0_0 = datetime.now()
    # timing shouldn't be a keyword or parameter
    timing = 0

    # Obs is a dictionary
    pol_names = obs['pol_names']
    dimension = int(obs['dimension'])
    elements = int(obs['elements'])
    kbinsize = obs['kpix']
    n_vis_arr = obs['nf_vis']
    # Units are in number of wavelengths
    kx_span = kbinsize * dimension
    ky_span = kx_span
    min_baseline = obs['min_baseline']
    max_baseline = obs['max_baseline']
    #interpolate_kernel in the psf dict should be Boolean
    if 'interpolate_kernel' in psf:
        interp_flag = psf['interpolate_kernel']
    else:
        interp_flag = False
    # IDL code has a pointer in the structure that holds a structure...assuming dict in a dict here.
    # The dict attached to the key baseline_info likely has to keys fbin_i, freq and freq_use
    freq_bin_i = obs['baseline_info']['fbin_i']
    #If nofi_use list provided
    if len(fi_use) == 0:
        #Get all non zeros from the list
        fi_use = np.nonzero(obs['baseline_info']['freq_use'])
    # Use the fi_use numpy array as indexes for freq_bin_i
    freq_bin_i = freq_bin_i[fi_use]
    n_freq = obs['n_freq']
        
    # Also what's the default for bi_use?
    if len(bi_use) == 0:
        # If the data is being gridded separately for the even/odd time samples, then force
        # flagging to be consistent across even/odd sets
            flag_test=np.sum(vis_weights, where=vis_weights > 0.1)
            bi_use = np.where(flag_test > 0, flag_test)
    else:
            b_info = obs['baseline_info']
            tile_use = np.nonzero(b_info['tile_use']) + 1
            # TODO: #array_match(b_info.tile_A,b_info.tile_B,value_match=tile_use)
            bi_use = None 

    n_b_use = len(bi_use)
    n_f_use = len(fi_use)

    # Calculate indices of visibilities to grid during this call (i.e. specific freqs, time sets)
    # and initialize output arrays
    # REMEMBER IDL does stores matrix as [columns, rows], while NumPy does [rows, columns]
    # So MATRIX_MULTIPLY(A, B) == A # B == B * A (in NumPy)
    vis_inds_use = np.ones(n_b_use) * fi_use + bi_use * np.ones(n_f_use) * n_freq
    #Check Above about vis_weight_switch, it could always be TRUE
    vis_weights_use = vis_weights[vis_inds_use]
    # Use only the parts we want
    vis_arr_use = visibility[vis_inds_use]
    # Get frequencies
    frequency_array = obs['baseline_info']['freq']
    frequency_array = frequency_array[fi_use]

    # We could save on memory use by not saving this as new variables
    complex_flag = psf['complex_flag']
    psf_dim = psf['dim']
    psf_resolution = psf['resolution']
    # Reform changes dimensions, in this case it should act like squeeze
    # polarization, freq_bin_i and bi_use might need to be an array or list
    group_arr = np.squeeze(psf['id'][polarization, freq_bin_i[fi_use], bi_use])
    # We may need a way to make python read a pointer here, or in other areas too.
    # Either that or we see if a dictionary can be used instead.
    beam_arr = psf['beam_ptr']
    
    uu = params['uu'][bi_use]
    vv = params['vv'][bi_use]
    ww = params['ww'][bi_use]
    kx_arr = uu/kbinsize
    ky_arr = vv /kbinsize

    nbaselines = obs['nbaselines']
    n_samples = obs['n_time']
    n_freq_use = len(frequency_array)
    psf_dim2 = 2 * psf_dim
    psf_dim3 = psf_dim * psf_dim
    bi_use_reduced = bi_use % nbaselines

    if beam_per_baseline:
        # Initialization for gridding operation via a low-res beam kernel, calculated per
        # baseline using offsets from image-space delays
        uv_grid_phase_only = True
        # What is this?
        # psf_intermediate_res=(Ceil(Sqrt(psf_resolution)/2)*2.)<psf_resolution
        # There is no < operator in IDL?
        # Operator from one of the libraries?
        psf_intermediate_res = np.ceil(np.sqrt(psf_resolution) / 2) *2
        psf_image_dim = psf['image_info']['psf_image_dim']
        image_bot = (psf_dim/2) * psf_intermediate_res + psf_image_dim/2
        image_top = (psf_dim*psf_resolution-1) - (psf_dim/2)*psf_intermediate_res + psf_image_dim/2
        # TODO: l_m_n(obs, psf, l_mode=l_mode, m_mode=m_mode)
        n_tracked = None 

        if uv_grid_phase_only:
            n_tracked = np.zeros_like(n_tracked)
        
    if grid_uniform:
        mapfn_recalculate = False
        
    if mapfn_recalculate:
        map_fn = None 
        #TODO Contains a pointer array of size dimension and elements
        # What's the alternative? NumPy 2D Array (probably)
    
    # Flag baselines on their maximum and minimum extent in the frequency range
    dist_test=np.sqrt((kx_arr)^2 + (ky_arr) ^ 2) * kbinsize
    dist_test_max = np.max(obs['baseline_info']['freq']) * dist_test
    dist_test_min = np.min(obs['baseline_info']['freq']) * dist_test
    flag_dis_baseline_min = np.where(dist_test_min < min_baseline, dist_test_min)
    flag_dis_baseline_max = np.where(dist_test_max > max_baseline, dist_test_max)
    flag_dis_baseline= np.concatenate(flag_dis_baseline_min, flag_dis_baseline_max)
    n_dist_flag = len(flag_dis_baseline)
    
    conj_i = np.where(ky_arr > 0, ky_arr)
    n_conj = len(conj_i)
    conj_flag = np.arange(len(ky_arr))
    if n_conj > 0:
        conj_flag[conj_i] = 1
        kx_arr[conj_i] = 0
        ky_arr[conj_i] = 0
        uu[conj_i] = 0
        vv[conj_i] = 0
        ww[conj_i] = 0
        vis_arr_use[:, conj_i] = np.conjugate(vis_arr_use[:, conj_i])
        if model_flag:
            model = model[:, conj_i] = np.conjugate(vis_arr_use[:, conj_i])
    
    # Center of baselines for x and y in units of pixels
    xcen = kx_arr * frequency_array
    ycen = ky_arr * frequency_array

    x = y = (np.arange(dimension) - (dimension / 2)) * obs['kpix']

    # Pixel number offset per baseline for each uv-box subset 
    x_offset = (np.floor((xcen - np.floor(xcen))*psf_resolution) % psf_resolution).astype(int)
    y_offset = (np.floor((ycen - np.floor(ycen))*psf_resolution) % psf_resolution).astype(int)
    # Derivatives from pixel edge to baseline center for use in interpolation
    dx_arr = (xcen - np.floor(xcen)) * psf_resolution - np.floor((xcen - np.floor(xcen)) * psf_resolution)
    dy_arr = (ycen - np.floor(ycen)) * psf_resolution - np.floor((ycen - np.floor(ycen)) * psf_resolution)
    

