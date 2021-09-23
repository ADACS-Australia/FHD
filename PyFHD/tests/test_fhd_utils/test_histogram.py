import pytest
import numpy as np
from fhd_utils.histogram import histogram
from scipy.io import readsav

def test_idl_example(shared_datadir) :
    """
    This test is based on the example from the IDL documentation.
    This ensures we get the same behaviour as an example everybody can see.
    """
    # Setup the test from the histogram data file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['idl_hist_example']
    hist, _, indices = histogram(data)
    # Set the expected
    expected_hist = testing_dict['idl_example_hist']
    expected_indices = testing_dict['idl_example_inds']
    # Assert they are equal element by element
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_hundred_nums(shared_datadir):
    """
    This is a basic test of an array with numbers 0 to 99 in increasing
    order.
    Should produce two bins.
    """
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['hundred_ints']
    hist, _, indices = histogram(data, bin_size = 50)
    expected_hist = testing_dict['hundred_ints_hist_bin50']
    expected_indices = testing_dict['hundred_ints_inds_bin50']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_hundred_ten_bins(shared_datadir):
     # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['hundred_ints']
    # This is to show that bin_size is ignored when num_bins is used
    hist, _, indices = histogram(data, num_bins = 10, bin_size=1000)
    expected_hist = testing_dict['hundred_ints_hist_nbin10']
    expected_indices = testing_dict['hundred_ints_inds_nbin10']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_min(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['hundred_ints']
    hist, _, indices = histogram(data, bin_size = 10, min = 10)
    expected_hist = testing_dict['hundred_ints_hist_min10']
    expected_indices = testing_dict['hundred_ints_inds_min10']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_max(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['hundred_ints']
    hist, _, indices = histogram(data, bin_size = 10, max = 50)
    expected_hist = testing_dict['hundred_ints_hist_max50']
    expected_indices = testing_dict['hundred_ints_inds_max50']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_min_max(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['hundred_ints']
    hist, _, indices = histogram(data, bin_size = 10, min = 10, max = 55)
    expected_hist = testing_dict['hundred_ints_hist_min10_max55']
    expected_indices = testing_dict['hundred_ints_inds_min10_max55']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_max(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['hundred_ints']
    hist, _, indices = histogram(data, max = 55)
    expected_hist = testing_dict['hundred_ints_hist_binsize1_max55']
    expected_indices = testing_dict['hundred_ints_inds_binsize1_max55']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['normals']
    hist, _, indices = histogram(data)
    expected_hist = testing_dict['normals_hist']
    expected_indices = testing_dict['normals_inds']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals_binsize(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['normals']
    hist, _, indices = histogram(data, bin_size = 0.25)
    expected_hist = testing_dict['normals_hist_binsize025']
    expected_indices = testing_dict['normals_inds_binsize025']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals_min_max(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['normals']
    hist, _, indices = histogram(data, min = 0, max = 1, bin_size = 0.25)
    expected_hist = testing_dict['normals_hist_min_max']
    expected_indices = testing_dict['normals_inds_binsize_min_max']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals(shared_datadir):
    # Read the histogram file
    testing_dict = readsav(shared_datadir / 'histogram.sav', python_dict=True)
    data = testing_dict['normals'] * 10
    hist, _, indices = histogram(data, bin_size = 2)
    expected_hist = testing_dict['normals_hist_times10']
    expected_indices = testing_dict['normals_inds_times10']
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

# TODO: Add more tests based on the data given by Nichole