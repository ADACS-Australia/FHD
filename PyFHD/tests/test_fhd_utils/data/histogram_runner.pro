PRO histogram_runner
    idl_hist_example = [[-5,  4,   2,  -8,  1], [ 3,  0,   5,  -5,  1], [ 6, -7,   4,  -4, -8], [-1, -5, -14,   2,  1]]

    idl_example_hist = histogram(IDL_hist_example, REVERSE_INDICES = idl_example_inds)

    hundred_ints = indgen(100)

    hundred_ints_hist_bin50=histogram(hundred_ints, binsize = 50, reverse_indices = hundred_ints_inds_bin50)

    hundred_ints_hist_nbin10 = histogram(hundred_ints, binsize = 10, reverse_indices = hundred_ints_inds_nbin10)

    hundred_ints_hist_min10 = histogram(hundred_ints, binsize = 10, min = 10, reverse_indices = hundred_ints_inds_min10)

    hundred_ints_hist_max50 = histogram(hundred_ints, binsize = 10, max = 50, reverse_indices = hundred_ints_inds_max50)

    hundred_ints_hist_min10_max55 = histogram(hundred_ints, binsize = 10, min = 10, max = 55, reverse_indices = hundred_ints_inds_min10_max55)

    hundred_ints_hist_binsize1_max55 = histogram(hundred_ints, binsize = 1,max = 55, reverse_indices = hundred_ints_inds_binsize1_max55)

    normals = RANDOMN(42, 10, 10)

    normals_hist = histogram(normals, reverse_indices = normals_inds)

    normals_hist_binsize025 = histogram(normals, binsize = 0.25, reverse_indices = normals_inds_binsize025)

    normals_hist_min_max = histogram(normals, binsize = 0.25, min = 0, max = 1, reverse_indices = normals_inds_binsize_min_max)

    normals_hist_times10 = histogram(normals*10, binsize = 2, reverse_indices = normals_inds_times10) 

    SAVE, /VARIABLES, FILENAME = 'histogram.sav'
END