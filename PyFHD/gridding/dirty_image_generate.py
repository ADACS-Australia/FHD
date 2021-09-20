import numpy as np
from fhd_utils.modified_astro.meshgrid import meshgrid

def dirty_image_generate(dirty_image_uv, mask = None, baseline_threshold = 0, normalization = None,
                         resize = None, width_smooth = None, degpix = None, no_real = False,
                         image_filter_fn = 'filter_uv_uniform', pad_uv_image = None, filter = None,
                         weights = None, beam_ptr = None, *args, **kwargs):
    pass