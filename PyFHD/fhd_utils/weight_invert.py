import numpy as np

def weight_invert(weights, threshold = None, abs = False):
    """
    The weights invert function cleans the weights given by removing
    the values that are 0, NaN or Inf ready for additional calculation.
    If a threshold is set, then we check the values that match the threshold
    instead of checking for zeros.

    Parameters
    ----------
    weights: array
        An array of values of some dtype
    threshold: int/float, optional
        A real number set as the threshold for the array.
        By default its set to None, in this case function checks
        for zeros.
    abs: bool, optional
        Set to False by default. When True, weights are compared against
        threshold or search for zeros using absolute values.

    Returns
    -------
    result: array
        The weights array that has had NaNs and Infinities removed, and zeros OR
        values that don't meet the threshold.
    """
    result = np.zeros(np.max(np.shape[0], 1), dtype = weights.dtype)
    # If we're told to use absolute values then create a copy of weights with absolute values
    if abs:
        weights_use = np.abs(weights)
    # Otherwise create a new variable which points to the same instance
    else:
        weights_use = weights
    # If threshold has been set then...
    if threshold is not None:
        # Get the indexes which meet the threshold
        i_use = np.where(weights_use >= threshold)
    else:
        # Otherwise get where they are not zero
        i_use = np.where(weights_use)
    if np.size(i_use) > 0:
            result[i_use] = 1 / weights[i_use]
    # Replace all NaNs with Zeros
    if np.size(np.where(np.isnan(result))) != 0:
        result[np.where(np.isnan(result))] = 0
    # Replace all Infinities with Zeros
    if np.size(np.where(np.isinf(result))) != 0:
        result[np.where(np.isinf(result))] = 0
    # If the result contains only 1 result, then return the result, not an array
    if np.size(result) == 1:
        result = result[0]
    return result
        
