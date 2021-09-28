from numpy.core.numeric import array_equal
import pytest
import numpy as np
from fhd_utils.rebin import rebin
from pathlib import Path

# TODO: Look at alternatives to python-testdir, it copies data to tempoerary dir for every test...

def get_data_expected(shared_datadir, function_name, data_filename, expected_filename):
    """
    This function is designed to read npy files in a 
    data directory inside fhd_utils. Ensure the data file
    has been made with the scripts inside the scripts directory.
    Use splitter.py to put the files and directories in the right format.
    Paths are expected to be of shared_datadir/data/function_name/[data,expected]_filename.npy
    shared_datadir is given by pytest-datadir, it should be the directory where the test file is in.

    Parameters
    ----------
    shared_datadir : Path
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
    input_path = Path(shared_datadir, function_name, data_filename)
    expected_path = Path(shared_datadir, function_name, expected_filename)
    input = np.load(input_path, allow_pickle=True)
    expected = np.load(expected_path, allow_pickle=True)
    # Return the input and expected
    return input, expected


def test_rebin_oneD_up(shared_datadir):
    """Testing rebin with using a 1D array and expanding it"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test.npy', 'test_1r_8c.npy')
    assert np.array_equal(rebin(input, (1,8)), expected)

def test_rebin_oneD_up2(shared_datadir):
    """Test with expanding to multiple rows and columns """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test.npy', 'test_2r_8c.npy')
    assert np.array_equal(rebin(input, (2,8)), expected)

def test_rebin_oneD_down(shared_datadir):
    """Testing rebin with using a 1D array and downscaling it"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test.npy', 'test_2c_1r.npy')
    assert np.array_equal(rebin(input, (1,2)), expected)
    
def test_rebin_oneD_down_up(shared_datadir):    
    """Testing same 1D but increasing in rows, going down in columns"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test.npy', 'test_2r_2c.npy')
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_oneD_extreme_down(shared_datadir):
    """testing same 1D but only wanting a single value"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test.npy', 'test_1r_1c.npy')
    assert np.array_equal(rebin(input, (1,1)), expected)

def test_rebin_oneD_same(shared_datadir):
    """testing same 1D but only wanting a single value"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test.npy', 'test_same.npy')
    assert np.array_equal(rebin(input, (1,4)), expected)

def test_rebin_vertical_array_up(shared_datadir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test2.npy', 'test2_vertical.npy')
    assert np.array_equal(rebin(input, (8, 1)), expected)

def test_rebin_vertical_array_to_square(shared_datadir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test2.npy', 'test2_to_square.npy')
    assert np.array_equal(rebin(input, (4, 4)), expected)

def test_rebin_vertical_array_to_rect(shared_datadir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test2.npy', 'test2_to_rect.npy')
    assert np.array_equal(rebin(input, (8, 4)), expected)   

def test_rebin_vertical_array_same(shared_datadir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'test2.npy', 'test2_same.npy')
    assert np.array_equal(rebin(input, (4, 1)), expected) 

def test_rebin_twoD_up_1_by_2(shared_datadir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data.npy', 'data_4r_10c.npy')
    assert np.array_equal(rebin(input, (4, 10)), expected)

def test_rebin_twoD_up_1_by_3(shared_datadir):
    """ Testing a 2D array only increasing columns by a factor of 3 """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data.npy', 'data_4r_15c.npy')
    assert np.array_equal(rebin(input, (4,15)), expected)

def test_rebin_twoD_up_2_by_2(shared_datadir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data.npy', 'data_8r_10c.npy')
    assert np.array_equal(rebin(input, (8, 10)), expected)

def test_rebin_twoD_same(shared_datadir):
    """Testing a 2D array by giving the same """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data.npy', 'data_same.npy')
    assert np.array_equal(rebin(input, (4, 5)), expected)

def test_rebin_twoD_down_2_by_3(shared_datadir):
    """ Testing a 2D Array but downscaling by a factor of 2 now """
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data2.npy', 'data2_2r_3c.npy')
    assert np.array_equal(rebin(input, (2, 3)), expected)

def test_rebin_twoD_down_2_by_2(shared_datadir):
    """Testing a 2D array downscaling to a small square"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data2.npy', 'data2_2r_2c.npy')    
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_twoD_down_in_half(shared_datadir):
    """Taking a 4x4aray and going to a square"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data3.npy', 'data3_2r_2c.npy')
    assert np.array_equal(rebin(input, (2,2)), expected)

def test_rebin_twoD_down_extreme(shared_datadir):
    """2D array into 1 value"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'data3.npy', 'data3_1r_1c.npy')
    assert np.array_equal(rebin(input, (1,1)), expected)

# Larger tests begin here

# EXPANDING

def test_rebin_hundred_100_by_100(shared_datadir):
    """Expand a 100 element 2D array to 100 x 100"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'hundred.npy', 'hundred_100r_100c.npy')
    assert np.array_equal(rebin(input, (100,100)), expected)

def test_rebin_hundred_1000_by_1000(shared_datadir):
    """Expand a 100 element 2D array to 1000 x 1000"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'hundred.npy', 'hundred_1kr_1kc.npy')
    assert np.array_equal(rebin(input, (1000,1000)), expected)

def test_rebin_hundred_billion(shared_datadir):
    """Expand a 100 element 2D array to a billion elements"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'hundred.npy', 'hundred_1e4r_1e5c.npy')
    assert np.array_equal(rebin(input, (1e4,1e5)), expected)

# DECREASING

def test_rebin_hundred_100_by_100(shared_datadir):
    """Take an array with a billion elements put it down into 100 x 100"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'billion.npy', 'billion_100r_100c.npy')
    assert np.array_equal(rebin(input, (100,100)), expected)

def test_rebin_hundred_1000_by_1000(shared_datadir):
    """Take an array with a billion elements put it down into 1000 x 1000"""
    input, expected = get_data_expected(shared_datadir, 'rebin', 'billion.npy', 'billion_1kr_1kc.npy')
    assert np.array_equal(rebin(input, (1000,1000)), expected)





