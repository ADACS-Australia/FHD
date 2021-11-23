from numpy.lib.function_base import interp
import pytest
import numpy as np
from pathlib import Path
from tests.test_utils import get_data, get_data_items
from fhd_utils.idl_tools.array_match import array_match
from fhd_utils.histogram import histogram
from fhd_core.gridding.baseline_grid_locations import baseline_grid_locations

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), '**/baseline_grid_locations/'))[0]

def test_baseline_one(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
        'input_psf_1.npy',
    )
    obs, params, vis_weights, fi_use, interp_flag = get_data_items(
        data_dir,
        'input_obs_1.npy',
        'input_params_1.npy',
        'input_vis_weight_ptr_1.npy',
        'input_fi_use_1.npy',
        'input_interp_flag_1.npy',
    )
    # Get the expected outputs
    expected_bin_n, expected_bin_i, expected_n_bin_use, expected_ri, expected_xmin,\
    expected_ymin, expected_vis_inds_use, expected_x_offset, expected_y_offset,\
    expected_dx0dy0, expected_dx0dy1, expected_dx1dy0, expected_dx1dy1 = get_data_items(
        data_dir,
        'output_bin_n_1.npy',
        'output_bin_i_1.npy',
        'output_n_bin_use_1.npy',
        'output_ri_1.npy',
        'output_xmin_1.npy',
        'output_ymin_1.npy',
        'output_vis_inds_use_1.npy',
        'output_x_offset_1.npy',
        'output_y_offset_1.npy',
        'output_dx0dy0_arr_1.npy',
        'output_dx0dy1_arr_1.npy',
        'output_dx1dy0_arr_1.npy',
        'output_dx1dy1_arr_1.npy',
    )
    # Use the baseline grid locations function
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, fi_use = fi_use, interp_flag = interp_flag)
    print(np.nonzero(np.abs(baselines_dict['ymin'] - expected_ymin)))
    # Check we got the right results from the dictionary
    assert np.array_equal(expected_vis_inds_use, baselines_dict['vis_inds_use'])
    assert np.array_equal(expected_x_offset, baselines_dict['x_offset'])
    assert np.array_equal(expected_y_offset, baselines_dict['y_offset'])
    assert np.array_equal(expected_xmin, baselines_dict['xmin'])
    assert np.array_equal(expected_ymin, baselines_dict['ymin'])
    assert np.array_equal(expected_bin_n, baselines_dict['bin_n'])
    assert np.array_equal(expected_bin_i, baselines_dict['bin_i'])
    assert expected_n_bin_use == baselines_dict['n_bin_use']
    assert np.array_equal(expected_ri, baselines_dict['ri'])
    assert np.array_equal(expected_dx0dy0, baselines_dict['dx0dy0_arr'])
    assert np.array_equal(expected_dx0dy1, baselines_dict['dx0dy1_arr'])
    assert np.array_equal(expected_dx1dy0, baselines_dict['dx1dy0_arr'])
    assert np.array_equal(expected_dx1dy1, baselines_dict['dx1dy1_arr'])

def test_baseline_two(data_dir):
    # Get the inputs
    obs, psf, params, vis_weights, fi_use, interp_flag, fill_model_vis = get_data_items(
        data_dir,
        'input_obs_2.npy',
        'input_psf_2.npy',
        'input_params_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_fi_use_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_fill_model_visibilities_2.npy'
    )
    # Get the expected outputs
    expected_bin_n, expected_bin_i, expected_n_bin_use, expected_ri, expected_xmin,\
    expected_ymin, expected_x_offset, expected_y_offset,\
    expected_dx0dy0, expected_dx0dy1, expected_dx1dy0, expected_dx1dy1 = get_data_items(
        data_dir,
        'output_bin_n_2.npy',
        'output_bin_i_2.npy',
        'output_n_bin_use_2.npy',
        'output_ri_2.npy',
        'output_xmin_2.npy',
        'output_ymin_2.npy',
        'output_x_offset_2.npy',
        'output_y_offset_2.npy',
        'output_dx0dy0_arr_2.npy',
        'output_dx0dy1_arr_2.npy',
        'output_dx1dy0_arr_2.npy',
        'output_dx1dy1_arr_2.npy',
    )
    # Use the baseline grid locations function
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, fi_use = fi_use, interp_flag = interp_flag, fill_model_visibilities = fill_model_vis)
    # Check we got the right results from the dictionary
    assert np.array_equal(expected_x_offset, baselines_dict['x_offset'])
    assert np.array_equal(expected_y_offset, baselines_dict['y_offset'])
    assert np.array_equal(expected_xmin, baselines_dict['xmin'])
    assert np.array_equal(expected_ymin, baselines_dict['ymin'])
    assert np.array_equal(expected_bin_n, baselines_dict['bin_n'])
    assert np.array_equal(expected_bin_i, baselines_dict['bin_i'])
    assert expected_n_bin_use == baselines_dict['n_bin_use']
    assert np.array_equal(expected_ri, baselines_dict['ri'])
    assert np.array_equal(expected_dx0dy0, baselines_dict['dx0dy0_arr'])
    assert np.array_equal(expected_dx0dy1, baselines_dict['dx0dy1_arr'])
    assert np.array_equal(expected_dx1dy0, baselines_dict['dx1dy0_arr'])
    assert np.array_equal(expected_dx1dy1, baselines_dict['dx1dy1_arr'])

def test_baseline_three(data_dir):
    # Get the inputs
    obs, psf, params, vis_weights, fi_use, interp_flag, bi_use = get_data_items(
        data_dir,
        'input_obs_3.npy',
        'input_psf_3.npy',
        'input_params_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_fi_use_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_bi_use_arr_3.npy'
    )
    # Get the expected outputs
    expected_bin_n, expected_bin_i, expected_n_bin_use, expected_ri, expected_xmin,\
    expected_ymin, expected_vis_inds_use, expected_x_offset, expected_y_offset,\
    expected_dx0dy0, expected_dx0dy1, expected_dx1dy0, expected_dx1dy1 = get_data_items(
        data_dir,
        'output_bin_n_3.npy',
        'output_bin_i_3.npy',
        'output_n_bin_use_3.npy',
        'output_ri_3.npy',
        'output_xmin_3.npy',
        'output_ymin_3.npy',
        'output_vis_inds_use_3.npy',
        'output_x_offset_3.npy',
        'output_y_offset_3.npy',
        'output_dx0dy0_arr_3.npy',
        'output_dx0dy1_arr_3.npy',
        'output_dx1dy0_arr_3.npy',
        'output_dx1dy1_arr_3.npy',
    )
    # Use the baseline grid locations function
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, fi_use = fi_use, interp_flag = interp_flag, bi_use = bi_use)
    # Check we got the right results from the dictionary
    assert np.array_equal(expected_vis_inds_use, baselines_dict['vis_inds_use'])
    assert np.array_equal(expected_x_offset, baselines_dict['x_offset'])
    assert np.array_equal(expected_y_offset, baselines_dict['y_offset'])
    assert np.array_equal(expected_xmin, baselines_dict['xmin'])
    assert np.array_equal(expected_ymin, baselines_dict['ymin'])
    assert np.array_equal(expected_bin_n, baselines_dict['bin_n'])
    assert np.array_equal(expected_bin_i, baselines_dict['bin_i'])
    assert expected_n_bin_use == baselines_dict['n_bin_use']
    assert np.array_equal(expected_ri, baselines_dict['ri'])
    assert np.array_equal(expected_dx0dy0, baselines_dict['dx0dy0_arr'])
    assert np.array_equal(expected_dx0dy1, baselines_dict['dx0dy1_arr'])
    assert np.array_equal(expected_dx1dy0, baselines_dict['dx1dy0_arr'])
    assert np.array_equal(expected_dx1dy1, baselines_dict['dx1dy1_arr'])

    