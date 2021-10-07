import numpy as np
from fhd_utils.modified_astro.meshgrid import meshgrid
from fhd_output.fft_filters.filter_uv_uniform import filter_uv_uniform
from fhd_utils.rebin import rebin
from math import sqrt
from astropy.convolution import convolve, Box2DKernel

def dirty_image_generate(dirty_image_uv, mask = None, baseline_threshold = 0, normalization = None,
                         resize = None, width_smooth = None, degpix = None, real = False,
                         image_filter_fn = 'filter_uv_uniform', pad_uv_image = None, filter = None,
                         weights = None, beam_ptr = None, *args, **kwargs):
    """
    TODO: Description

    Parameters
    ----------
    dirty_image_uv: array
        The array of which a dirty image will be constructed
    mask: array, optional
        A mask that is to be applied, of same size as dirty_image_uv.
        Default is None.
    baseline_threshold: int, optional
        Default is 0, threshold for boxed mask
    normalization: array, optional
        Default is None, provide an array same size as dirty_image_uv, or if resize
        is used, normalization should be the same size after resizing
    resize: int, float, optional
        Default is None. If set, triggers a resize routine that resizes by that factor
    width_smooth: int, float, optional
        If left as None defaults to np.floor(sqrt(dimension * elements) / 100),
        elements, dimension = dirty_image_uv.shape
    degpix: int, float, optional
        Apply the degree transformation to the dirty image
    real: bool, optional
        If True returns the real part only at the end
    image_filter_fn: String, optional
        By default is 'filter_uv_uniform', but can also be:
        filter_uv_hanning
        filter_uv_natural
        filter_uv_optimal
        filter_uv_radial
        filter_uv_tapered_uniform
    pad_uv_image: int, float, optional
        By default is None, if set, then will pad dirty image by said factor
    filter: array, optional
        By default is none, if set and is the same size as dirty_image_uv, then
        the filter will be directly applied, otherwise, its used as the filter input
        for image_filter_fn
    weights: array, optional
        In the case of using a filter, these are the weights for the filter function
    beam_ptr: array
        By default None, in the case filter_uv_optimal was used, this array will be
        applied
    
    Returns
    -------
    dirty_image: array
        A dirty image that has been transformed, filtered and maybe normalized.
    """
    # dimension is columns, elements is rows
    elements, dimension = dirty_image_uv.shape
    di_uv_use = dirty_image_uv
    # If width smooth hasn't been set, set it
    if width_smooth is None:
        width_smooth = np.floor(sqrt(dimension * elements) / 100)
    rarray = np.sqrt((meshgrid(dimension, 1) - dimension / 2) ** 2 + (meshgrid(elements, 2) - elements / 2) ** 2)
    # Get all the values that meet the threshold
    if baseline_threshold >= 0:
        cut_i = np.where(rarray < baseline_threshold)
    else:
        cut_i = np.where(rarray > np.abs(baseline_threshold))
    # Createthe mask array of ones
    mask_bt = np.ones((elements, dimension))
    # If there are values from cut, then use all those here and replace with 0
    if np.size(cut_i) > 0:
        mask_bt[cut_i] = 0
    # Use a box width averaging filter over the mask
    mask_bt = convolve(mask_bt, Box2DKernel(width_smooth > 1))    
    # Apply boxed mask to the dirty image
    di_uv_use *=  mask_bt
    # If a mask was supplied use that too
    if mask is not None:
        di_uv_use *= mask
    # If a filter was supplied as a numpy array (we can adjust this to support different formats)
    if filter is not None:
        if isinstance(np.ndarray):
            # If the filter is already the right size, use it
            if np.size(filter) == np.size(di_uv_use):
                di_uv_use *= filter
            # Otherwise use a filter function
            else:
                di_uv_use = eval("{}(di_uv_use, weights, filter)".format(image_filter_fn))
    # Resize the dirty image by the factor resize    
    if resize is not None:
        dimension *= resize
        elements *= resize
        di_uv_real = di_uv_use.real
        di_uv_img = di_uv_use.imag
        # Use rebin to resize, apply to real and complex separately
        di_uv_real = rebin(di_uv_real, (elements, dimension))
        di_uv_img = rebin(di_uv_img, (elements, dimension))
        # Combine real and complex back together
        di_uv_use = di_uv_real + di_uv_img * 1j
    #Apply padding if it was supplied
    if pad_uv_image is not None:
        dimension_new = np.max((np.max(dimension, elements) * pad_uv_image), np.max(dimension, elements))
        di_uv1 = np.zeros((dimension_new, dimension_new), dtype = "complex")
        di_uv1[dimension_new / 2 - elements / 2 : dimension_new / 2 + elements / 2 - 1,
               dimension_new / 2 - dimension / 2 : dimension_new / 2 + dimension / 2 - 1]
        di_uv_use = di_uv1 * (pad_uv_image ** 2)
    
    # FFT normalization
    if degpix is not None:
        di_uv_use /= np.radians(degpix) ** 2

    # Multivariate Fast Fourier Transform
    dirty_image = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(di_uv_use)))
    if real:
        dirty_image = dirty_image.real
    
    # filter_uv_optimal produces images that are weighted by one factor of the beam
    # Weight by an additional factor of the beam to align with FHD's convention
    if image_filter_fn == 'filter_uv_optimal' and beam_ptr is not None:
        dirty_image *= beam_ptr
    
    # If we are returning complex, make sure its complex
    if real:
        dirty_image = dirty_image.real
    else:
        dirty_image = dirty_image.astype("complex")
    # Normalize by the matrix given, if it was given
    if normalization is not None:
        dirty_image *= normalization
    #Return
    return dirty_image  