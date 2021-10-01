import numpy as np
from pathlib import Path

def get_data_expected(data_dir, data_filename, expected_filename, *args):
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
    expected_path = Path(data_dir, expected_filename)
    input = np.load(input_path, allow_pickle=True)
    expected = np.load(expected_path, allow_pickle=True)
    return_list = [input, expected]
    if len(args) > 0:
        for file in args:
            path = Path(data_dir, file)
            output = np.load(path, allow_pickle=True)
            return_list.append(output)
    # Return the input and expected
    return return_list