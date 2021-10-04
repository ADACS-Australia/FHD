import numpy as np
from pathlib import Path
from tests.test_utils import get_data_expected
from fhd_utils.idl_tools.array_match import array_match
import pytest

@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return list(Path.glob(Path.cwd(), '**/array_match/'))[0]

def test_array_match_uno(data_dir):
    array1, array2, value_match, expected_indices, expected_n_match =  get_data_expected(data_dir, 
                                                                       'vis_weights_update_input_array1.npy', 
                                                                       'vis_weights_update_input_array2.npy', 
                                                                       'vis_weights_update_input_value_match.npy',
                                                                       'vis_weights_update_output_match_indices.npy',
                                                                       'vis_weights_update_output_n_match.npy')
    # The files given contain structured numpy arrays, retrieve the values from them.
    array1 = array1.item().get('array1')
    array2 = array2.item().get('array2')
    value_match = value_match.item().get('value_match')
    expected_indices = expected_indices.item().get('match_indices')
    expected_n_match = expected_n_match.item().get('n_match')
    # Get the result and see if they match.
    indices, n_match = array_match(array1, value_match, array_2 = array2)
    assert indices == expected_indices
    assert n_match == expected_n_match