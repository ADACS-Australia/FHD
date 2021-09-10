import numpy as np
import datetime

from numpy.testing._private.utils import print_assert_equal
from fhd_utils.array_match import array_match
from fhd_utils.l_m_n import l_m_n
from fhd_utils.histogram import histogram
import interpolate_kernel

def visibility_grid(visibility, vis_weights, obs, status_str, psf, params,
                    file_path_fhd= "/.", weights_flag = False, variance_flag = False, polarization = 0,
                    map_flag = False, uniform_flag = False, fi_use = [],
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
    # Initalize uv-arrays
    image_uv = np.zeros((elements, dimension), dtype = complex)
    weights = np.zeros((elements, dimension), dtype = complex)
    variance = np.zeros((elements, dimension))
    uniform_filter = np.zeros((elements, dimension))

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
            bi_use = array_match(b_info['tile_A'], tile_use, array_2=b_info['tile_B'])

    n_b_use = len(bi_use)
    n_f_use = len(fi_use)

    # Calculate indices of visibilities to grid during this call (i.e. specific freqs, time sets)
    # and initialize output arrays
    # REMEMBER IDL does stores matrix as [columns, rows], while NumPy does [rows, columns]
    # So MATRIX_MULTIPLY(A, B) == A # B == B * A (in NumPy)
    vis_inds_use = np.ones_like(n_b_use) * fi_use + bi_use * np.ones_like(n_f_use) * n_freq
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
        # The < operator in IDL takes the minimum between two comparisons
        psf_intermediate_res = np.min(np.ceil(np.sqrt(psf_resolution) / 2) *2, psf_resolution)
        psf_image_dim = psf['image_info']['psf_image_dim']
        image_bot = (psf_dim/2) * psf_intermediate_res + psf_image_dim/2
        image_top = (psf_dim*psf_resolution-1) - (psf_dim/2)*psf_intermediate_res + psf_image_dim/2
        l_mode, m_mode, n_tracked = l_m_n(obs, psf)

        if uv_grid_phase_only:
            n_tracked = n_tracked * 0
    
    if grid_uniform:
        map_flag = False
    
    # Flag baselines on their maximum and minimum extent in the frequency range
    dist_test=np.sqrt((kx_arr)^2 + (ky_arr) ^ 2) * kbinsize
    dist_test_max = np.max(obs['baseline_info']['freq']) * dist_test
    dist_test_min = np.min(obs['baseline_info']['freq']) * dist_test
    flag_dist_baseline = np.where((dist_test_min < min_baseline) | (dist_test_max > max_baseline))
    n_dist_flag = len(flag_dist_baseline)
    
    conj_i = np.where(ky_arr > 0, ky_arr)
    n_conj = len(conj_i)
    conj_flag = np.zeros(len(ky_arr))
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
    dx0dy0_arr = (1 - dx_arr) * (1 - dy_arr)
    dx0dy1_arr = (1 - dx_arr) * dy_arr
    dx1dy0_arr = dx_arr * (1 - dy_arr)
    dx1dy1_arr = dy_arr * dx_arr
    # The minimum pixel in the uv-grid (bottom left of the kernel) that each baseline contributes to
    xmin = np.floor(xcen) + dimension / 2 - (psf_dim/2 - 1)
    ymin = np.floor(ycen) + elements / 2 - (psf_dim/2 - 1)

    # Set the minimum pixel value of baselines which fall outside of the uv-grid to -1 to exclude them
    range_test_x_i = np.where((xmin < 0) | ((xmin + psf_dim - 1) >= dimension - 1))
    range_test_y_i = np.where((ymin < 0) | ((ymin + psf_dim - 1) >= elements - 1))
    if (range_test_x_i.shape[0] * range_test_x_i.shape[1]) > 0:
        xmin[range_test_x_i] = ymin[range_test_x_i] = -1
    if (range_test_y_i.shape[0] * range_test_y_i.shape[1]) > 0:
        xmin[range_test_y_i] = ymin[range_test_y_i] = -1
    
    if n_dist_flag > 0:
        # If baselines fall outside the desired min/max baseline range at all during the frequency range, 
        # then set their minimum pixel value to -1 to exclude them 
        xmin[:, flag_dist_baseline] = -1
        ymin[: flag_dist_baseline] = -1

    # If baselines are flagged via the weights, then set their minimum pixel value to -1 to exclude them
    flag_i = np.where(vis_weights < 0)
    vis_weights = 0
    if flag_i.shape[0] * flag_i.shape[1] > 0:
        xmin[flag_i] = -1
        ymin[flag_i] = -1
    flag_i = 0

    if mask_mirror_indices:
        # Option to exclude v-axis mirrored baselines
        if n_conj > 0:
            xmin[:, conj_i] = -1
            ymin[:, conj_i] = -1
    
    if np.min(np.max(xmin), np.max(ymin)) < 0:
        # Return if all baselines have been flagged
        print("All data flagged or cut!")
        timing = datetime.now() - t0_0
        image_uv = np.zeros((elements, dimension), dtype=complex)
        # There are values like n_vis and error and timing which are
        # set but not returned as they are done as parameters in the function
        # May need to adjust the return statement to reflect this
        # Return might need to be a dictionary
        return image_uv 
    
    # Match all visibilities that map from and to exactly the same pixels and store them as a histogram in bin_n
    # with their respective index ri. Setting min equal to 0 excludes flagged (i.e. (xmin,ymin)=(-1,-1)) data
    bin_n, _, ri = histogram(xmin + ymin * dimension, min = 0)
    # This will get all the values that are not 0
    bin_i = np.where(bin_n)
    n_bin_use = bin_i.shape[0] * bin_i.shape[1]

    ind_ref = np.arange(np.max(bin_n))
    n_vis = np.sum(bin_n)
    for fi in range(n_f_use - 1):
        n_vis_arr[fi_use[fi]] = np.sum(xmin[fi, :], where = xmin[fi, :] > 0)
    # obs['nf_vis'] = n_vis_arr I'm guessing we'll be returning obs too...

    index_arr = np.reshape(np.arange(dimension * elements), [elements, dimension])
    if complex_flag:
        init_arr = np.zeros((psf_dim2, psf_dim2), dtype = complex)
    else:
        init_arr = np.zeros((psf_dim2, psf_dim2))
    
    arr_type = init_arr.dtype
    if grid_spectral:
        spectral_A = np.zeros((elements, dimension), dtype = complex)
        spectral_B = spectral_D = np.zeros((elements, dimension))
    if model_flag:
        spectral_model_A = spectral_A
    
    if map_flag:
        map_fn = np.zeros((elements, dimension))
        #Initialize ONLY those elements of the map_fn array that will receive data to remain sparse
        for bi in range(n_bin_use - 1):
            xmin1 = xmin[ri[ri[bin_i[bi]]]]
            ymin1 = ymin[ri[ri[bin_i[bi]]]]
            inds_init = np.zeros_like(map_fn[xmin1:xmin1+psf_dim-1,ymin1:ymin1+psf_dim-1])
            inds_init = index_arr[xmin1:xmin1+psf_dim-1,ymin1:ymin1+psf_dim-1][inds_init]
            nzero = np.size(inds_init)
            for ii in range(nzero-1):
                map_fn[inds_init[ii]] = init_arr
        map_fn_inds = np.zeros((psf_dim, psf_dim))
        psf2_inds = np.reshape(np.arange(psf_dim2 * psf_dim2), [psf_dim2, psf_dim2])
        for i in range(psf_dim - 1):
            for j in range(psf_dim - 1):
                map_fn_inds[i, j] = psf2_inds[psf_dim-i : 2 * psf_dim - i - 1, psf_dim - j : 2 * psf_dim - j - 1]
    
    for bi in range(n_bin_use - 1):
        # Cycle through sets of visibilities which contribute to the same data/model uv-plane pixels, and perform
        # the gridding operation per set using each visibilities' hyperresolved kernel

        # Select the indices of the visibilities which contribute to the same data/model uv-plane pixels
        inds = ri[ri[bin_i[bi]]:ri[bin_i[bi]+1]-1]
        ind0 = inds[0]

        # Select the pixel offsets of the hyperresolution uv-kernel of the selected visibilities 
        x_off = x_offset[inds]
        y_off = y_offset[inds]

        # Since all selected visibilities have the same minimum x,y pixel they contribute to, reduce the array
        xmin_use = xmin[ind0]
        ymin_use = ymin[ind0]

        # Find the frequency group per index
        freq_i = inds % n_freq_use
        fbin = freq_bin_i[freq_i]

        # Calculate the number of selected visibilities and their baseline index
        vis_n = bin_n[bin_i[bi]]
        baseline_inds = bi_use_reduced[ (inds / n_f_use) % nbaselines]

        if interp_flag:
            # Calculate the interpolated kernel on the uv-grid given the derivatives to baseline locations
            # and the hyperresolved pre-calculated beam kernel

            # Select the 2D derivatives to baseline locations
            dx1dy1 = dx1dy1_arr[inds]
            dx1dy0 = dx1dy0_arr[inds]
            dx0dy1 = dx0dy1_arr[inds]
            dx0dy0 = dx0dy0_arr[inds]

            # Select the model/data visibility values of the set, each with a weight of 1
            rep_flag = False
            if model_flag:
                model_box = model[inds]
            vis_box = vis_arr_use[inds]
            psf_weight = np.ones(vis_n)

            box_matrix = np.empty((vis_n, psf_dim3), dtype = arr_type)
            for ii in range(vis_n - 1):
                # For each visibility, calculate the kernel values on the static uv-grid given the
                # hyperresolved kernel and an interpolation involving the derivatives
                box_matrix[psf_dim3*ii] = interpolate_kernel(beam_arr[polarization, fbin[ii], baseline_inds[ii]], x_off[ii], y_off[ii], 
                                                             dx0dy0[ii], dx1dy0[ii], dx0dy1[ii], dx1dy1[ii])
        else:
            # Calculate the beam kernel at each baseline location given the hyperresolved pre-calculated beam kernel

            # Calculate a unique index for each kernel location and kernel type in order to reduce 
            # operations if there are repeats
            group_id = group_arr[inds]
            group_max = np.max(group_id) + 1
            xyf_i = (x_off + y_off * psf_resolution + fbin * psf_resolution ** 2) * group_max + group_id

            # Calculate the unique number of kernel locations/types
            xyf_si = np.sort(xyf_i)
            xyf_i = xyf_i[xyf_si]
            xyf_ui = np.unique(xyf_i)
            n_xyf_bin = np.size(xyf_ui)

            # There might be a better selection criteria to determine which is most efficient
            # If there is one, do you want to implement it here?
            if vis_n > 1.1 * n_xyf_bin and beam_per_baseline:
                # If there are any baselines which use the same beam kernel and the same discretized location
                # given the hyperresolution, then reduce the number of gridding operations to only
                # non-repeated baselines
                rep_flag = True
                inds = inds[xyf_si]
                inds_use = xyf_si[xyf_ui]
                freq_i = freq_i[inds_use]

                x_off = x_off[inds_use]
                y_off = y_off[inds_use]
                fbin = fbin[inds_use]
                baseline_inds = baseline_inds[inds_use]
                if n_xyf_bin > 1:
                    xyf_ui0 = [0, xyf_ui[0:n_xyf_bin-2] + 1]
                else:
                    xyf_ui0 = 0
                psf_weight = xyf_ui - xyf_ui0 + 1

                vis_box1 = vis_arr_use[inds]
                vis_box = vis_box1[xyf_ui]
                if model_flag:
                    model_box1 = model[inds]
                    model_box = model_box1[xyf_ui]
                
                # For the baselines which map to the same pixels and use the same beam,
                # add the underlying data/model pixels such that the gridding operation
                # only needs to be performed once for the set
                repeat_i = np.where(psf_weight > 1)

                xyf_ui = xyf_ui[repeat_i]
                xyf_ui0 = xyf_ui0[repeat_i]
                for rep_ii in range(np.size(repeat_i) - 1):
                    vis_box = [repeat_i[rep_ii]] = np.sum(model_box1[xyf_ui0[rep_ii]:xyf_ui[rep_ii]])
                vis_n = n_xyf_bin
            else:
                # If there are not enough baselines which use the same beam kernel and discretized
                # location to warrent reduction, then perform the gridding operation per baseline
                rep_flag = False
                if model_flag:
                    model_box = model[inds]
                    vis_box = vis_arr_use[inds]
                    psf_weight = np.ones(vis_n)
                    bt_index = inds / n_freq_use
            
            box_matrix = np.empty((vis_n, psf_dim3), dtype = arr_type)
            if beam_per_baseline:
                #  Make the beams on the fly with corrective phases given the baseline location for each visibility
                # to the static uv-grid
                # TODO: grid_beam_per_baseline
                #box_matrix = grid_beam_per_baseline()
                pass
            else:
                for ii in range(vis_n - 1):
                    # For each visibility, calculate the kernel values on the static uv-grid given the
                    # hyperresolved kernel
                    # Was dereferencing into two pointers
                    box_matrix[psf_dim3 * ii] = beam_arr[polarization,fbin[ii],baseline_inds[ii]][x_off[ii],y_off[ii]]

        # Calculate the conjugate transpose (dagger) of the uv-pixels that the current beam kernel contributes to
        if complex_flag:
            box_matrix_dag = np.conjugate(box_matrix)
        else:
            box_matrix_dag = box_matrix.real
        if map_flag and rep_flag:
            pass
            # TODO: Rebin Function in Python
            # box_matrix *= rebin(np.transpose(psf_weight), psf_dim3, vis_n)\
        
        if grid_spectral:
            term_A_box = np.transpose(box_matrix_dag) * np.transpose(freq_i * vis_box / n_vis)
            term_B_box = np.transpose(box_matrix_dag) * np.transpose(freq_i / n_vis)
            term_D_box = np.transpose(box_matrix_dag) * np.transpose(freq_i ** 2 / n_vis)

            spectral_A[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += term_A_box
            spectral_B[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += term_B_box
            spectral_D[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += term_D_box
            if model_flag:
                term_Am_box = np.transpose(box_matrix_dag) * np.transpose(model_box / n_vis)
                spectral_model_A[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += term_Am_box

        if model_flag:
            # If model visibilities are being gridded, calculate the product of the model vis and the beam kernel
            # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane
            box_arr = np.transpose(box_matrix_dag) * np.transpose(model_box / n_vis)
            model[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += box_arr
        
        # Calculate the product of the data vis and the beam kernel
        # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane
        box_arr = np.transpose(box_matrix_dag) * np.transpose(vis_box / n_vis)
        image_uv[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += box_arr

        if weights_flag:
            wts_box = np.transpose(box_matrix_dag) * np.transpose(psf_weight / n_vis)
            weights[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += wts_box
        if variance_flag:
            var_box = np.transpose(np.abs(box_matrix_dag) ** 2) * np.transpose(psf_weight / n_vis)
            variance[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1] += var_box
        if uniform_flag:
            uniform_filter[xmin_use:xmin_use+psf_dim-1,ymin_use:ymin_use+psf_dim-1]+=bin_n[bin_i[bi]]
        if map_flag:
            # If the mapping function is being calculated, then calculate the beam mapping for the current
            # set of uv-pixels and add to the full mapping function
            box_arr_map = np.transpose(box_matrix_dag) * box_matrix
            for i in range(psf_dim - 1):
                for j in range(psf_dim - 1):
                    ij = i + j * psf_dim
                    # Will need adjusting probably as its using arrays of pointers
                    map_fn[xmin_use+i,ymin_use+j][map_fn_inds[i,j]] += box_arr_map[:,ij]
    
    # They do things after the loop has finished to free memory
    # We may have to, but also Python usually handles itself

    if map_flag:
        
        

