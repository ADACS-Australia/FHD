import numpy as np

def interpolate_kernel(kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1):
    """
    TODO: Description

    Parameters
    ----------
    kernel_arr: array
        TODO: Description
    x_offset: array
        TODO: Description
    y_offset: array
        TODO: Description
    dx0dy0: array
        TODO: Description
    dx1dy0: array
        TODO: Description
    dx0dy1: array
        TODO: Description
    dx1dy1: array
        TODO: Description

    Returns
    -------
    kernel: array
        TODO: Description
    """
    # Potential issue could arise as this may need the values to swap around...
    kernel = kernel_arr[x_offset, y_offset] * dx0dy0
    kernel += kernel_arr[x_offset + 1, y_offset] * dx1dy0
    kernel += kernel_arr[x_offset, y_offset + 1] * dx0dy1
    kernel += kernel_arr[x_offset + 1, y_offset + 1] * dx1dy1

    return kernel