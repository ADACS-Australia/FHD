import numpy as np
from pathlib import Path

def get_data_expected(data_dir, data_filename, *args):
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
    *args : List
        If given, is expected to be more filenames
    
    Returns
    -------
    input : 
        The data used for input of the function being tested
    expected : 
        The expected result of the function
    """
    # Put as Paths and read the files
    input_path = Path(data_dir, data_filename)
    input = np.load(input_path, allow_pickle=True)
    return_list = [input]
    if len(args) > 0:
        for file in args:
            path = Path(data_dir, file)
            output = np.load(path, allow_pickle=True)
            return_list.append(output)
    # Return the input and expected
    return return_list

def process_inputs(data_dir, kernel_arr_path, x_offset_path, y_offset_path, dx0dy0_path, dx1dy0_path, dx0dy1_path, dx1dy1_path, expected_kernel_path):
    """
    Takes all the path inputs from interpolate_kernel tests and processes them so they're ready for use.

    Parameters
    ----------
    data_dir : Path
        Path to the interpolate_kernel data directory
    kernel_arr_path : Path
        filename for kernel_arr
    x_offset_path : PathPath
        filename for x_offset
    y_offset_path : Path
        filename for y_offset
    dx0dy0_path : Path
        filename for dx0dy0
    dx1dy0_path : Path
        filename for dx1dy0
    dx0dy1_path : Path
        filename for dx0dy1
    dx1dy1_path : Path
        filename for dx1dy1
    expected_kernel_path : Path
        filename for the output

    Returns
    -------
    kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1, expected_kernel
        Variables required to do the test
    """
    # Retrieve the files and their contents
    dx0dy0, dx1dy0, dx0dy1, dx1dy1, \
    x_offset, y_offset, kernel_arr,\
    expected_kernel = get_data_expected(data_dir,\
        dx0dy0_path, dx1dy0_path, dx0dy1_path, \
        dx1dy1_path, x_offset_path, y_offset_path,\
        kernel_arr_path, expected_kernel_path
    )
    # Get the items we want
    dx0dy0 = dx0dy0.item().get('dx0dy0')
    dx1dy0 = dx1dy0.item().get('dx1dy0')
    dx0dy1 = dx0dy1.item().get('dx0dy1')
    dx1dy1 = dx1dy1.item().get('dx1dy1')
    x_offset = x_offset.item().get('x_offset')
    y_offset = y_offset.item().get('y_offset')
    kernel_arr = kernel_arr.item().get('input')
    expected_kernel = expected_kernel.item().get('output')
    #Return them
    return kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1, expected_kernel