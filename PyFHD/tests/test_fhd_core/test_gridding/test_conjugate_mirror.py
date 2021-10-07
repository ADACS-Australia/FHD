import pytest
import numpy as np
from pathlib import Path
from fhd_core.gridding.conjugate_mirror import conjugate_mirror
from tests.test_utils import get_data_sav

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), "**/conjugate_mirror"))[0]

def test_conj_mirror_one(data_dir):
    input, expected_image = get_data_sav(data_dir, 
                                     'visibility_grid_input_1.sav',
                                     'visibility_grid_output_1.sav')
    image = conjugate_mirror(input)
    print("Input")
    print(input)
    print(input.dtype)
    print(np.nonzero(input))
    print("Image")
    print(image)
    print(image.dtype)
    print(np.nonzero(image))
    print("Expected")
    print(expected_image)
    print(expected_image.dtype)
    print(np.nonzero(expected_image))
    print("Maximum Difference")
    print(np.max(image - expected_image))
    print(np.max(input - expected_image))
    assert np.array_equal(image, expected_image)

def test_conj_mirror_two(data_dir):
    input, expected_image = get_data_sav(data_dir, 
                                     'visibility_grid_input_2.sav',
                                     'visibility_grid_output_2.sav')
    image = conjugate_mirror(input)
    print("Input")
    print(input)
    print(input.dtype)
    print(np.nonzero(input))
    print("Image")
    print(image)
    print(image.dtype)
    print(np.nonzero(image))
    print("Expected")
    print(expected_image)
    print(expected_image.dtype)
    print(np.nonzero(expected_image))
    print("Maximum Difference")
    print(np.max(image - expected_image))
    assert np.array_equal(image, expected_image)

def test_conj_mirror_three(data_dir):
    input, expected_image = get_data_sav(data_dir, 
                                     'visibility_grid_input_3.sav',
                                     'visibility_grid_output_3.sav')
    image = conjugate_mirror(input)
    print("Input")
    print(input)
    print(input.dtype)
    print(np.nonzero(input))
    print("Image")
    print(image)
    print(image.dtype)
    print(np.nonzero(image))
    print("Expected")
    print(expected_image)
    print(expected_image.dtype)
    print(np.nonzero(expected_image))
    print("Maximum Difference")
    print(np.max(image - expected_image))
    assert np.array_equal(image, expected_image)