from numpy.core.numeric import array_equal
import pytest
import numpy as np
from fhd_utils.rebin import rebin
from pathlib import Path

def get_data_expected(data_dir, data_filename, expected_filename):
    """
    This function is designed to read npy files in a 
    data directory inside fhd_utils. Ensure the data file
    has been made with the scripts inside the scripts directory.
    Use splitter.py to put the files and directories in the right format.
    Paths are expected to be of data_dir/data/function_name/[data,expected]_filename.npy
    data_dir is given by pytest-datadir, it should be the directory where the test file is in.

    Parameters
    ----------
    data_dir : Path
        This should be the dir passed through from pytest-datadir
    function_name : String
        The name of the function we're testing
    data_filename : String
        The name of the file for the input
    expected_filename : String
        The name of the file name for the expected result
    
    Returns
    -------
    input : 
        The data used for input of the function being tested
    expected : 
        The expected result of the function
    """
    # Put as Paths and read the files
    input_path = Path(data_dir, data_filename)
    expected_path = Path(data_dir, expected_filename)
    input = np.load(input_path, allow_pickle=True)
    expected = np.load(expected_path, allow_pickle=True)
    # Return the input and expected
    return input, expected

@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return list(Path.glob(Path.cwd(), '**/rebin/'))[0]

def test_rebin_oneD_up(data_dir):
    """Testing rebin with using a 1D array and expanding it"""
    input, expected = get_data_expected(data_dir, 'test.npy', 'test_1r_8c.npy')
    assert np.array_equal(rebin(input, (1,8)), expected)

def test_rebin_oneD_up2(data_dir):
    """Test with expanding to multiple rows and columns """
    input, expected = get_data_expected(data_dir, 'test.npy', 'test_2r_8c.npy')
    assert np.array_equal(rebin(input, (2,8)), expected)

def test_rebin_oneD_down(data_dir):
    """Testing rebin with using a 1D array and downscaling it"""
    input, expected = get_data_expected(data_dir, 'test.npy', 'test_2c_1r.npy')
    assert np.array_equal(rebin(input, (1,2)), expected)
    
def test_rebin_oneD_down_up(data_dir):    
    """Testing same 1D but increasing in rows, going down in columns"""
    input, expected = get_data_expected(data_dir, 'test.npy', 'test_2r_2c.npy')
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_oneD_extreme_down(data_dir):
    """testing same 1D but only wanting a single value"""
    input, expected = get_data_expected(data_dir, 'test.npy', 'test_1r_1c.npy')
    assert np.array_equal(rebin(input, (1,1)), expected)

def test_rebin_oneD_same(data_dir):
    """testing same 1D but only wanting a single value"""
    input, expected = get_data_expected(data_dir, 'test.npy', 'test_same.npy')
    assert np.array_equal(rebin(input, (1,4)), expected)

def test_rebin_vertical_array_up(data_dir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    input, expected = get_data_expected(data_dir, 'test2.npy', 'test2_vertical.npy')
    assert np.array_equal(rebin(input, (8, 1)), expected)

def test_rebin_vertical_array_to_square(data_dir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    input, expected = get_data_expected(data_dir, 'test2.npy', 'test2_to_square.npy')
    assert np.array_equal(rebin(input, (4, 4)), expected)

def test_rebin_vertical_array_to_smaller_square(data_dir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    input, expected = get_data_expected(data_dir, 'test2.npy', 'test2_to_smaller_square.npy')
    assert np.array_equal(rebin(input, (2, 2)), expected) 

def test_rebin_vertical_array_to_rect(data_dir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    input, expected = get_data_expected(data_dir, 'test2.npy', 'test2_to_rect.npy')
    assert np.array_equal(rebin(input, (8, 4)), expected)   

def test_rebin_vertical_array_to_smaller_rect(data_dir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    input, expected = get_data_expected(data_dir, 'test2.npy', 'test2_to_smaller_rect.npy')
    assert np.array_equal(rebin(input, (2, 1)), expected)   

def test_rebin_vertical_array_same(data_dir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    input, expected = get_data_expected(data_dir, 'test2.npy', 'test2_same.npy')
    assert np.array_equal(rebin(input, (4, 1)), expected) 

def test_rebin_twoD_up_1_by_2(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_4r_10c.npy')
    assert np.array_equal(rebin(input, (4, 10)), expected)

def test_rebin_twoD_up_1_by_3(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 3 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_4r_15c.npy')
    assert np.array_equal(rebin(input, (4,15)), expected)

def test_rebin_increase_rows_only():
    input = np.array([[2,6,9],[8,20,18], [15, 16, 19]])
    expected = np.array([[2,6,9], [4,10,12],[6,15,15], [8,20,18], [10,18,18],[12,17,18], [15, 16, 19], [15, 16, 19], [15, 16, 19]])
    assert np.array_equal(rebin(input, (9,3)), expected)

def test_rebin_basic():
    """Testing rebin with using a 1D array and expanding it"""
    input = np.array([[2,5],[6,7]])
    expected= np.array([[2,3,5,5],[4,5,6,6],[6,6,7,7],[6,6,7,7]])
    assert np.array_equal(rebin(input, (4,4)), expected)

def test_rebin_twoD_up_2_by_2(data_dir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_8r_10c.npy')
    assert np.array_equal(rebin(input, (8, 10)), expected)

def test_rebin_twoD_up_3_by_2(data_dir):
    """ Testing a 2D array increasing rows and columns by factors of 3 and 2 respectively"""
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_12r_10c.npy')
    assert np.array_equal(rebin(input, (12, 10)), expected)

def test_rebin_twoD_up_2_by_3(data_dir):
    """ Testing a 2D array increasing rows and columns by factors of 2 and 3 respectively """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_8r_15c.npy')
    assert np.array_equal(rebin(input, (8, 15)), expected)

def test_rebin_twoD_same(data_dir):
    """Testing a 2D array by giving the same """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_same.npy')
    assert np.array_equal(rebin(input, (4, 5)), expected)

def test_rebin_twoD_down_2_by_3(data_dir):
    """ Testing a 2D Array but downscaling by a factor of 2 now """
    input, expected = get_data_expected(data_dir, 'data2.npy', 'data2_2r_3c.npy')
    assert np.array_equal(rebin(input, (2, 3)), expected)

def test_rebin_twoD_down_2_by_2(data_dir):
    """Testing a 2D array downscaling to a small square"""
    input, expected = get_data_expected(data_dir, 'data2.npy', 'data2_2r_2c.npy')    
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_twoD_down_in_half(data_dir):
    """Taking a 4x4aray and going to a square"""
    input, expected = get_data_expected(data_dir, 'data3.npy', 'data3_2r_2c.npy')
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_twoD_down_extreme(data_dir):
    """2D array into 1 value"""
    input, expected = get_data_expected(data_dir, 'data3.npy', 'data3_1r_1c.npy')
    assert np.array_equal(rebin(input, (1,1)), expected)

def test_rebin_all_zeros_expand():
    input = np.zeros((2,2))
    expected = np.zeros((6,6))
    assert np.array_equal(rebin(input, (6,6)), expected)

def test_rebin_all_zeros_down():
    input = np.zeros((6,6))
    expected = np.zeros((2,2))
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_all_ones_expand():
    input = np.ones((3,3))
    expected = np.ones((9,9))
    assert np.array_equal(rebin(input, (9,9)), expected)

# Tests for Floats

def test_rebin_fl_up_rows(data_dir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_8r_5c.npy')
    assert np.array_equal(rebin(input, (8, 5)), expected)

def test_rebin_fl_up_cols(data_dir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_4r_10c.npy')
    assert np.array_equal(rebin(input, (4, 10)), expected)

def test_rebin_fl_up_2_by_2(data_dir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_8r_10c.npy')
    assert np.array_equal(rebin(input, (8, 10)), expected)

def test_rebin_fl_down_rows_cols(data_dir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_2r_1c.npy')
    assert np.array_equal(rebin(input, (2, 1)), expected)

def test_rebin_fl_down_rows(data_dir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_2r_5c.npy')
    assert np.array_equal(rebin(input, (2, 5)), expected)

# Larger tests begin here

# EXPANDING

def test_rebin_twoD_20_rows(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_20r.npy')
    assert np.array_equal(rebin(input, (20, 5)), expected)

def test_rebin_twoD_20_columns(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_20c.npy')
    assert np.array_equal(rebin(input, (4, 20)), expected)

def test_rebin_twoD_20_rows_20_columns(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_20r_20c.npy')
    assert np.array_equal(rebin(input, (20, 20)), expected)

def test_rebin_twoD_50_columns(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_50c.npy')
    assert np.array_equal(rebin(input, (4, 50)), expected)

def test_rebin_twoD_40_rows(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_40r.npy')
    assert np.array_equal(rebin(input, (40, 5)), expected)

def test_rebin_twoD_2000(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data.npy', 'data_2000.npy')
    assert np.array_equal(rebin(input, (40, 50)), expected)

def test_rebin_hundred_10_by_100(data_dir):
    """Expand a 100 element 2D array to 10 x 100"""
    input, expected = get_data_expected(data_dir, 'hundred.npy', 'hundred_10r_100c.npy')
    assert np.array_equal(rebin(input, (10,100)), expected)

def test_rebin_hundred_100_by_10(data_dir):
    """Expand a 100 element 2D array to 100 x 10"""
    input, expected = get_data_expected(data_dir, 'hundred.npy', 'hundred_100r_10c.npy')
    assert np.array_equal(rebin(input, (100,10)), expected)

def test_rebin_hundred_100_by_100(data_dir):
    """Expand a 100 element 2D array to 100 x 100"""
    input, expected = get_data_expected(data_dir, 'hundred.npy', 'hundred_100r_100c.npy')
    assert np.array_equal(rebin(input, (100,100)), expected)

def test_rebin_hundred_1000_by_1000(data_dir):
    """Expand a 100 element 2D array to 1000 x 1000"""
    input, expected = get_data_expected(data_dir, 'hundred.npy', 'hundred_1kr_1kc.npy')
    assert np.array_equal(rebin(input, (1000,1000)), expected)

def test_rebin_hundred_billion(data_dir):
    """Expand a 100 element 2D array to a billion elements"""
    input, expected = get_data_expected(data_dir, 'hundred.npy', 'hundred_1e4r_1e5c.npy')
    assert np.array_equal(rebin(input, (1e4,1e5)), expected)

# DECREASING

def test_rebin_billion_100_by_100(data_dir):
    """Take an array with a billion elements put it down into 100 x 100"""
    input, expected = get_data_expected(data_dir, 'billion.npy', 'billion_100r_100c.npy')
    assert np.array_equal(rebin(input, (100,100)), expected)

def test_rebin_billion_1000_by_1000(data_dir):
    """Take an array with a billion elements put it down into 1000 x 1000"""
    input, expected = get_data_expected(data_dir, 'billion.npy', 'billion_1kr_1kc.npy')
    assert np.array_equal(rebin(input, (1000,1000)), expected)

def test_rebin_billion_to_1(data_dir):
    """Take an array with a billion elements put it down into 1000 x 1000"""
    input, expected = get_data_expected(data_dir, 'billion.npy', 'billion_extreme.npy')
    assert np.array_equal(rebin(input, (1,1)), expected)

# Float Large

def test_rebin_fl_20_rows(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_20r.npy')
    assert np.array_equal(rebin(input, (20, 5)), expected)

def test_rebin_fl_20_columns(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_20c.npy')
    assert np.array_equal(rebin(input, (4, 20)), expected)

def test_rebin_fl_20_rows_20_columns(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_20r_20c.npy')
    assert np.array_equal(rebin(input, (20, 20)), expected)

def test_rebin_fl_50_columns(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_50c.npy')
    assert np.array_equal(rebin(input, (4, 50)), expected)

def test_rebin_fl_40_rows(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_40r.npy')
    assert np.array_equal(rebin(input, (40, 5)), expected)

def test_rebin_fl_2000(data_dir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(data_dir, 'data_fl.npy', 'data_fl_2000.npy')
    assert np.array_equal(rebin(input, (40, 50)), expected)





