from numpy.core.numeric import array_equal
import pytest
import numpy as np
from fhd_utils.rebin import rebin

# TESTING VARIABLES
test = np.array([0,10,20,30])
test2 = np.array([[0], [10],[20],[30]])
data = np.array([[ -5,   4,   2,  -8,   1],
                 [  3,   0,   5,  -5,   1],
                 [  6,  -7,   4,  -4,  -8],
                 [ -1,  -5, -14,   2,   1]])
data2 = np.array([[ -5,   4,   2,  -8,   1,  4],
                  [  3,   0,   5,  -5,   1,  4],
                  [  6,  -7,   4,  -4,  -8,  3],
                  [ -1,  -5, -14,   2,   1,  8]])
data3 = np.array([[5,-4,8,0],[9,10,20,2],[1,0,1,3],[15,-12,5,4]])

def test_rebin_oneD_up():
    # Testing rebin with using a 1D array and expanding it
    expected = np.array([ 0,  5, 10, 15, 20, 25, 30, 30])
    assert np.array_equal(rebin(test, (1,8)), expected)

def test_rebin_oneD_up2():
    # Test with expanding to multiple rows and columns 
    expected = np.array([[ 0,  5, 10, 15, 20, 25, 30, 30],
                         [ 0,  5, 10, 15, 20, 25, 30, 30]])
    assert np.array_equal(rebin(test, (2,8)), expected)

def test_rebin_oneD_down():
    # Testing rebin with using a 1D array and downscaling it
    expected = np.array([5, 25])
    assert np.array_equal(rebin(test, (1,2)), expected)
    
def test_rebin_oneD_down_up():    
    # Testing same 1D but increasing in rows, going down in columns
    expected = np.array([[5,25],[5,25]])
    assert np.array_equal(rebin(test, (2,2)), expected)

def test_rebin_oneD_extreme_down():
    # testing same 1D but only wanting a single value
    expected = np.array([15])
    assert np.array_equal(rebin(test, (1,1)), expected)

def test_rebin_oneD_vertical():
    # Testing a 1D array that's vertical (i.e. shape of (x, 1))
    expected = np.array([[0],[5],[10],[15],[20],[25],[30],[30]])
    assert np.array_equal(rebin(test2, (8, 1)), expected)

def test_rebin_twoD_up_1_by_2():
    # Testing a 2D array only increasing columns by a factor of 2
    expected = np.array([[ -4,   0,   3,   2,   1,  -3,  -7,  -3,   1,   1],
                         [  2,   1,   0,   2,   4,   0,  -4,  -1,   1,   1],
                         [  5,   0,  -6,  -1,   3,   0,  -4,  -6,  -8,  -8],
                         [ -1,  -3,  -5,  -9, -13,  -5,   1,   1,   1,   1]])
    assert np.array_equal(rebin(data, (4, 10)), expected)

def test_rebin_twoD_up_1_by_3():
    # Testing a 2D array only increasing columns by a factor of 3
    expected = np.array([[ -4,  -1,   1,   3,   3,   2,   1,  -1,  -4,  -7,  -4,  -1,  1,  1,  1],
                         [  2,   1,   0,   0,   1,   3,   4,   1,  -1,  -4,  -2,  0,  1,  1,  1],
                         [  5,   1,  -2,  -6,  -3,   0,   3,   1,  -1,  -4,  -5,  -6,  -8,  -8,  -8],
                         [ -1,  -2,  -3,  -5,  -8, -11, -13,  -8,  -3,   1,   1,   1,   1,  1,   1]])
    assert np.array_equal(rebin(data, (4,15)), expected)

def test_rebin_twoD_up_2_by_2():
    # Testing a 2D array increasing both rows and columns by factors of 2
    expected = np.array([[ -3,   0,   2,   2,   1,  -2,  -6,  -2,   1,   1],
                         [  0,   0,   1,   2,   2,  -1,  -5,  -1,   1,   1],
                         [  2,   0,   0,   1,   3,   0,  -4,  -1,   0,   0],
                         [  3,   0,  -3,   0,   3,   0,  -4,  -3,  -3,  -3],
                         [  4,   0,  -5,  -1,   2,   0,  -3,  -5,  -7,  -7],
                         [  1,  -1,  -5,  -5,  -5,  -2,  -1,  -2,  -3,  -3],
                         [ -1,  -3,  -5,  -9, -13,  -5,   1,   1,   1,   1],
                         [ -1,  -3,  -5,  -9, -13,  -5,   1,   1,   1,   1]])
    assert np.array_equal(rebin(data, (8, 10)), expected)

def test_rebin_twoD_down_2_by_3():
    expected = np.array([[ 0,  -1,  2],
                         [-1,  -3,  1]])
    assert np.array_equal(rebin(data2, (2, 3)), expected)

def test_rebin_twoD_down_2_by_2():
    expected = np.array([[  1,  0],
                         [ -2,  0]])
    assert np.array_equal(rebin(data2, (2,2)), expected)

def test_rebin_twoD_down_in_half():
    expected = np.array([[ 4,  7], 
                         [ 0,  3]])
    assert np.array_equal(rebin(data3, (2,2)), expected)


