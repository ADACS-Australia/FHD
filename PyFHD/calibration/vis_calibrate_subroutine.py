import numpy as np
import warnings
import calculate_adaptive_gain

def vis_calibrate_subroutine(vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, preserve_visibilities = False, 
                             calibration_weights = False,  no_ref_tile = True):
    """[summary]

    Parameters
    ----------
    vis_ptr : [type]
        [description]
    vis_model_ptr : [type]
        [description]
    vis_weight_ptr : [type]
        [description]
    obs : [type]
        [description]
    cal : [type]
        [description]
    preserve_visibilities : bool, optional
        [description], by default False
    calibration_weights : bool, optional
        [description], by default False
    no_ref_tile : bool, optional
        [description], by default True
    """
    # Retrieve values from data structures
    reference_tile = cal['ref_antenna']
    min_baseline = obs['min_baseline']
    max_baseline = obs['max_baseline']
    dimension = obs['dimension']
    elements = obs['elements']
    min_cal_baseline = cal['min_cal_baseline']
    max_cal_baseline = cal['max_cal_baseline']
    # minimum number of calibration equations needed to solve for the gain of one baseline
    min_cal_solutions = cal['min_solns']
    # average the visibilities across time steps before solving for the gains
    time_average = cal['time_avg']
    # maximum iterations to perform for the linear least-squares solver
    max_cal_iter = cal['max_iter']
    # Leave a warning if its less than 5 iterations
    if max_cal_iter < 5:
        warnings.warn("At Least 5 calibrations iterations is recommended.\nYou're currently using {} iterations".format(int(max_cal_iter)))
    conv_thresh = cal['conv_thresh']
    use_adaptive_gain = cal['adaptive_gain']
    base_gain = cal['base_gain']
    # halt if the strict convergence is worse than most of the last x iterations
    divergence_history = 3
    # halt if the convergence gets significantly worse by a factor of x in one iteration
    divergence_factor = 1.5
    n_pol = cal['n_pol']
    n_freq = cal['n_freq']
    n_tile = cal['n_tile']
    n_time = cal['n_time']
    # weights WILL be over-written! (Only for NAN gain solutions)
    vis_weight_ptr_use = vis_weight_ptr
    # tile_A & tile_B contribution indexed from 0
    tile_A_i = cal['tile_A'] - 1
    tile_A_i = cal['tile_B'] - 1
    freq_arr = cal['freq']
    bin_offset = cal['bin_offset']
    n_baselines = obs['n_baselines']
    if 'phase_iter' in cal.dtype.names:
        phase_fit_iter = cal['phase_iter']
    else:
        phase_fit_iter = np.min(np.floor(max_cal_iter / 4), 4)
    kbinsize = obs['kpix']
    cal_return = cal

    for pol_i in range(n_pol):
        convergence = np.zeros(n_tile, n_freq)
        conv_iter_arr = np.zeros(n_tile, n_freq)
        gain_arr = cal['gain'][pol_i]

        # Average the visibilities over the time steps before solving for the gains solutions
        # This is not recommended, as longer baselines will be downweighted artifically.
        if time_average:
            # The visibilities have dimension nfreq x (n_baselines x n_time),
            # which can be reformed to nfreq x n_baselines x n_time
            tile_A_i = tile_A_i[0 : n_baselines]
            tile_B_i = tile_B_i[0 : n_baselines]
            # So IDL does reforms as REFORM(x, cols, rows, num_of_col_row_arrays)
            # Python is row-major, so we need to flip that shape that is used in REFORM
            shape = np.flip(np.array([n_freq, n_baselines, n_time]))
            vis_weight_use = np.min(np.max(0, np.reshape(vis_weight_ptr_use[pol_i], shape)), 1)
            vis_model = np.reshape(vis_model_ptr[pol_i], shape)
            vis_model = np.sum(vis_model * vis_weight_use, axis = 0)
            vis_measured = np.reshape(vis_ptr[pol_i], shape)
            vis_avg = np.sum(vis_measured * vis_weight_use, axis = 0)
            weight = np.sum(vis_weight_use, axis = 0)

            kx_arr = cal['uu'][0 : n_baselines] / kbinsize
            ky_arr = cal['vv'][0 : n_baselines] / kbinsize
            kr_arr = np.sqrt(kx_arr ** 2 + ky_arr ** 2)
            dist_arr = np.dot(kr_arr, freq_arr) * kbinsize
            xcen = np.dot(abs(kx_arr), freq_arr)
            ycen = np.dot(abs(ky_arr), freq_arr)

            if calibration_weights:
                flag_dist_cut = np.where((dist_arr < min_baseline) | (xcen > (elements / 2)) | (ycen > (dimension / 2))
                if min_cal_baseline > min_baseline:
                    taper_min = np.max((np.sqrt(2) * min_cal_baseline - dist_arr) / min_cal_baseline, 0)
                else:
                    taper_min = 0
                if max_cal_baseline < max_baseline:
                    taper_max = np.max((dist_arr - max_cal_baseline) / min_cal_baseline, 0)
                else:
                    taper_max = 0
                baseline_weights = np.max(1 - (taper_min + taper_max) ** 2, 0)
            else:
                flag_dist_cut = np.where((dist_arr < min_cal_baseline) | (dist_arr > max_cal_baseline) | (xcen > elements / 2) | (ycen > dimension / 2))
        else:
            vis_weight_use = np.max()

