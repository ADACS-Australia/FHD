import numpy as np

def meshgrid(dimension, elements, return_float = False):
    """
    Generates a 2D array of X or Y values. Could be replaced by a another function 
    
    Parameters
    ----------
    dimension: int
        Sets the size of the array to return
    elements: int
        Sets the size of the array to return
    return_float: bool, optional
        The default is False, dtype is implied by dimension and/or elements
        by default. If True, sets return array to float.
    
    Returns
    -------
    result: ndarray
        A numpy array of shape (elements, dimension) that is 
        a modified np.arange(elements * dimension).
    """
    # If elements is set as 2
    if elements == 2:
        # Replicate LINDGEN by doing arange and resizing, result is floored
        result = np.reshape(np.floor(np.arange(elements * dimension) / dimension), (elements, dimension))
    else:
        # Replicate LINDGEN by doing arange and resizing 
        result = np.reshape(np.arange(elements * dimension) % dimension, (elements, dimension))
    # If we need to return a float, then set the result and return
    if return_float:
        return result.astype("float")
    # Otherwise return as is
    else:
        return result