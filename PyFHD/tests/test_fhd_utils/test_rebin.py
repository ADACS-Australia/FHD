from numpy.core.numeric import array_equal
import pytest
import numpy as np
from fhd_utils.rebin import rebin
from scipy.io import readsav

'''
This test is designed to read a datafile in a 
data directory inside fhd_utils. Ensure the data file
has been made with the scripts inside the scripts directory.
The file that has the correct tests are rebin_runner.pro
'''

def test_rebin_oneD_up(shared_datadir):
    """Testing rebin with using a 1D array and expanding it"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test']
    expected = testing_dict['test_1r_8c']
    assert np.array_equal(rebin(test, (1,8)), expected)

def test_rebin_oneD_up2(shared_datadir):
    """Test with expanding to multiple rows and columns """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test']
    expected = testing_dict['test_2r_8c']
    assert np.array_equal(rebin(test, (2,8)), expected)

def test_rebin_oneD_down(shared_datadir):
    """Testing rebin with using a 1D array and downscaling it"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test']
    expected = testing_dict['test_2c_1r']
    assert np.array_equal(rebin(test, (1,2)), expected)
    
def test_rebin_oneD_down_up(shared_datadir):    
    """Testing same 1D but increasing in rows, going down in columns"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test']
    expected = testing_dict['test_2r_2c']
    assert np.array_equal(rebin(test, (2,2)), expected)

def test_rebin_oneD_extreme_down(shared_datadir):
    """testing same 1D but only wanting a single value"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test']
    expected = testing_dict['test_1r_1c']
    assert np.array_equal(rebin(test, (1,1)), expected)

def test_rebin_oneD_same(shared_datadir):
    """testing same 1D but only wanting a single value"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test']
    expected = testing_dict['test_same']
    assert np.array_equal(rebin(test, (1,4)), expected)

def test_rebin_vertical_array_up(shared_datadir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test2']
    expected = testing_dict['test2_vertical']
    assert np.array_equal(rebin(test, (8, 1)), expected)

def test_rebin_vertical_array_to_square(shared_datadir):
    """Testing a 1D array that's vertical (i.e. shape of (x, 1))"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test2']
    expected = testing_dict['test2_to_square']
    assert np.array_equal(rebin(test, (4, 4)), expected)

def test_rebin_vertical_array_to_rect(shared_datadir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test2']
    expected = testing_dict['test2_to_rect']
    assert np.array_equal(rebin(test, (8, 4)), expected)   

def test_rebin_vertical_array_same(shared_datadir):
    """ Testing a 1D array that's vertical (i.e. shape of (x, 1)) """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['test2']
    expected = testing_dict['test2_same']
    assert np.array_equal(rebin(test, (4, 1)), expected) 

def test_rebin_twoD_up_1_by_2(shared_datadir):
    """ Testing a 2D array only increasing columns by a factor of 2 """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data']
    expected = testing_dict['data_4r_10c']
    assert np.array_equal(rebin(test, (4, 10)), expected)

def test_rebin_twoD_up_1_by_3(shared_datadir):
    """ Testing a 2D array only increasing columns by a factor of 3 """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data']
    expected = testing_dict['data_4r_15c']
    assert np.array_equal(rebin(test, (4,15)), expected)

def test_rebin_twoD_up_2_by_2(shared_datadir):
    """ Testing a 2D array increasing both rows and columns by factors of 2 """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data']
    expected = testing_dict['data_8r_10c']
    assert np.array_equal(rebin(test, (8, 10)), expected)

def test_rebin_twoD_same(shared_datadir):
    """Testing a 2D array by giving the same """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data']
    expected = testing_dict['data_same']
    assert np.array_equal(rebin(test, (4, 5)), expected)

def test_rebin_twoD_down_2_by_3(shared_datadir):
    """ Testing a 2D Array but downscaling by a factor of 2 now """
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data2']
    expected = testing_dict['data2_2r_3c']
    assert np.array_equal(rebin(test, (2, 3)), expected)

def test_rebin_twoD_down_2_by_2(shared_datadir):
    """Testing a 2D array downscaling to a small square"""    
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data2']
    expected = testing_dict['data2_2r_2c']
    assert np.array_equal(rebin(test, (2,2)), expected)

def test_rebin_twoD_down_in_half(shared_datadir):
    """Taking a 4x4aray and going to a square"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data3']
    expected = testing_dict['data3_2r_2c']
    assert np.array_equal(rebin(test, (2,2)), expected)

def test_rebin_twoD_down_extreme(shared_datadir):
    """2D array into 1 value"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['data3']
    expected = testing_dict['data3_1r_1c']
    assert np.array_equal(rebin(test, (1,1)), expected)

# Larger tests begin here

# EXPANDING

def test_rebin_hundred_100_by_100(shared_datadir):
    """Expand a 100 element 2D array to 100 x 100"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['hundred']
    expected = testing_dict['hundred_100r_100c']
    assert np.array_equal(rebin(test, (100,100)), expected)

def test_rebin_hundred_1000_by_1000(shared_datadir):
    """Expand a 100 element 2D array to 1000 x 1000"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['hundred']
    expected = testing_dict['hundred_1kr_1kc']
    assert np.array_equal(rebin(test, (1000,1000)), expected)

def test_rebin_hundred_billion(shared_datadir):
    """Expand a 100 element 2D array to a billion elements"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['hundred']
    expected = testing_dict['hundred_1e4r_1e5c']
    assert np.array_equal(rebin(test, (1e4,1e5)), expected)

# DECREASING

def test_rebin_hundred_100_by_100(shared_datadir):
    """Take an array with a billion elements put it down into 100 x 100"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['billion']
    expected = testing_dict['billion_100r_100c']
    assert np.array_equal(rebin(test, (100,100)), expected)

def test_rebin_hundred_1000_by_1000(shared_datadir):
    """Take an array with a billion elements put it down into 1000 x 1000"""
    testing_dict = readsav(shared_datadir / 'rebin.sav', python_dict=True)
    test = testing_dict['billion']
    expected = testing_dict['billion_1kr_1kc']
    assert np.array_equal(rebin(test, (1000,1000)), expected)





