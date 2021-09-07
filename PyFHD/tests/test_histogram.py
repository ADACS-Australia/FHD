import pytest
import numpy as np
from fhd_utils.histogram import histogram

def test_idl_example() :
    """
    This test is based on the example from the IDL documentation.
    This ensures we get the same behaviour as an example everybody can see.
    """
    # Setup the test
    data = np.array([[ -5,   4,   2,  -8,   1],
                    [  3,   0,   5,  -5,   1],
                    [  6,  -7,   4,  -4,  -8],
                    [ -1,  -5, -14,   2,   1]])

    hist, indices = histogram(data)
    # Set the expected
    expected_hist = np.array([1, 0, 0, 0, 0, 0, 2, 1, 0, 3, 1, 0, 0, 1, 1, 3, 2, 1, 2, 1, 1]) 
    expected_indices = np.array([22, 23, 23, 23, 23, 23, 23, 25, 26, 26, 29, 30, 30, 30, 31, 32, 35,
                              37, 38, 40, 41, 42, 17,  3, 14, 11,  0,  8, 16, 13, 15,  6,  4,  9,
                              19,  2, 18,  5,  1, 12,  7, 10])
    # Assert they are equal element by element
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_hundred_nums():
    """
    This is a basic test of an array with numbers 0 to 99 in increasing
    order.
    Should produce two bins.
    """
    data = np.arange(100)
    hist, indices = histogram(data, bin_size = 50)
    expected_hist = np.array([50, 50])
    expected_indices = np.array([3, 53, 103] + list(range(100)))
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_hundred_ten_bins():
    data = np.arange(100)
    # This is to show that bin_size is ignored when num_bins is used
    hist, indices = histogram(data, num_bins = 10, bin_size=1000)
    expected_hist = np.ones(10) * 10
    expected_indices = np.array(list(range(11,121, 10)) + list(range(100)))
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_min():
    data = np.arange(100)
    # This is to show that bin_size is ignored when num_bins is used
    hist, indices = histogram(data, bin_size = 10, min = 10)
    expected_hist = np.ones(9) * 10
    expected_indices = np.array(list(range(10,110, 10)) + list(range(10,100)))
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_max():
    data = np.arange(100)
    # This is to show that bin_size is ignored when num_bins is used
    hist, indices = histogram(data, bin_size = 10, max = 50)
    expected_hist = np.append(np.ones(5) * 10, 1)
    expected_indices = np.array(list(range(7,67, 10)) + [57] + list(range(0,51)))
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_min_max():
    data = np.arange(100)
    # This is to show that bin_size is ignored when num_bins is used
    hist, indices = histogram(data, bin_size = 10, min = 10, max = 55)
    expected_hist = np.append(np.ones(4) * 10, 6)
    expected_indices = np.array(list(range(5, 55, 10)) + list(range(10,56)))
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)