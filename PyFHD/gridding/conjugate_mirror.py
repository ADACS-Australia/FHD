import numpy as np

def conjugate_mirror(image):
    """
    This takes a 2D array and mirrors it, shifts it and
    its an array of complex numbers its get the conjugates
    of the 2D array

    Parameters
    ----------
    image: array
        A 2D array of real or complex numbers
    
    Returns
    -------
    conj_mirror_image: array
        The mirrored and shifted image array
    """
    # Yes all of this could have been done in fewer lines, but that's hard to read
    # Flip image left to right (i.e. flips columns)
    conj_mirror_image = np.fliplr(image)
    # Flip image up to down (i.e. flips rows)
    conj_mirror_image = np.flipud(image)
    # Shifts rows and columns by 1 on each axis
    conj_mirror_image = np.roll(conj_mirror_image , (1, 1), axis = (0,1))
    # If any of the array is complex, or its a complex array, get the conjugates
    if np.iscomplexobj(image):   
        conj_mirror_image = np.conjugate(conj_mirror_image)
    return conj_mirror_image