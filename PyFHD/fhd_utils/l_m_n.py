import numpy as np

def l_m_n(obs, psf, obsdec = None, obsra = None, dec_arr = None, ra_arr = None) :
    """
    Calculates the l mode, m mode and the n_tracked
    TODO: Add Detailed Description of l_m_n

    Parameters
    ----------
    obs: dict

    psf: dict

    obsdec: array, optional
        By default is set to None, as such by default this value will be set to
        obs['obsdec']
    obsra: array, optional
        By default is set to None, as such by default this value will be set to
        obs['obsra']
    dec_arr: array, optional
        By default is set to None, as such by default this value will be set to
        psf['image_info']['dec_arr']
    ra_arr: array, optional
        By default is set to None, as such by default this value will be set to
        psf['image_info']['ra_arr']

    Returns
    -------
    l_mode: array
        TODO: Add description for l_mode
    m_mode: array
        TODO: Add description for m_mode
    n_tracked: array
        TODO: Add description for n_tracked.
    """
    # If the variables passed through are None them
    if obsdec is None:
        obsdec = obs['obsdec']
    if obsra is None:
        obsra = obs['obsra']
    if dec_arr is None:
        declination_arr = psf['image_info']['dec_arr']
    if ra_arr  is None:
        right_ascension_arr = psf['image_info']['ra_arr']

    # Convert all the degrees given into radians
    obsdec = np.radians(obsdec)
    obsra = np.radians(obsra)
    dec_arr = np.radians(declination_arr)
    ra_arr = np.radians(right_ascension_arr)

    # Calculate l mode, m mode and the phase-tracked n mode of pixel centers
    cdec0 = np.cos(obsdec)
    sdec0 = np.sin(obsdec)
    cdec = np.cos(declination_arr)
    sdec = np.sin(declination_arr)
    cdra = np.cos(right_ascension_arr - obsra)
    sdra = np.sin(right_ascension_arr - obsra)
    l_mode = cdec * sdra
    m_mode = sdec * cdec0 - cdec * sdec0 * cdra
    # n=1 at phase center, so reference from there for phase tracking
    n_tracked = (sdec * sdec0 - cdec * cdec0 * cdra) - 1

    # find any NaN values
    nan_vals = np.where(np.isnan(n_tracked))
    # If any found, replace them with 0's
    if np.size(nan_vals) > 0:
        n_tracked[nan_vals] = 0
        l_mode[nan_vals] = 0
        m_mode[nan_vals] = 0
    
    # Return the modes
    return l_mode, m_mode, n_tracked

    