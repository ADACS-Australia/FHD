import math

"""
TODO:  Doctest
"""
def l_m_n(obs, psf, obsdec = False, obsra = False, dec_arr = None, ra_arr = None) :
    """
    TODO: Docstring
    """
    if obsdec:
        obsdec = obs['obsdec']
    if obsra:
        obsra = obs['obsra']