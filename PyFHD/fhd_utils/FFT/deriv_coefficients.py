import numpy as np
import math

def deriv_coefficients(n, divide_factorial = False):
    """
    Computes an array of coefficients resulting in taking the 
    n-th derivative of a function of the form x^a (a must not 
    be a positive integer less than n)

    Parameters
    ----------
    n: int
        Decides the length of coefficients
    divide_factorial: bool
        Determine if we need to divide by the factorial

    Returns
    -------
    coeff: ndarray
        An array of coefficients
    """
    if n <= 0:
        raise ValueError("n must be > 0 to set an array size")
    # Set up the array
    coeff = np.zeros(n)
    # Set the first coefficient to 1
    coeff[0] = 1
    # For every coefficient
    for m in range(1, n):
        coeff[1:m] += -m * coeff[0: m-1]
    # If we are to divide by the factorial do that to each coefficient
    if divide_factorial:
        for m in range(n):
            coeff[m] /= math.factorial(m + 1)
    
    # Return coefficients
    return coeff


    
