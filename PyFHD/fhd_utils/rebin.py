import numpy as np

def rebin(a, shape):
    """
    Resizes a 2d array by averaging or repeating elements, 
    new dimensions must be integral factors of original dimensions
    Parameters
    ----------
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array
        
    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array, the data are averaged, 
        if the new shape is bigger array elements are repeated

    Examples
    --------
    

    References
    ----------
    [1] https://stackoverflow.com/a/8090605
    """
    old_shape  = a.shape
    if len(old_shape) == 1:
        old_shape = (1,old_shape[0])
    # If we are downsizing
    if shape[0] < old_shape[0]:
        if old_shape[0] % shape[0] != 0 or old_shape[1] % shape[1] != 0:
            raise ValueError("Your new shape should be a factor of the original shape")
        sh = shape[0], old_shape[0] // shape[0], shape[1], old_shape[1] // shape[1]
        rebinned = np.reshape(a, sh)
        rebinned = rebinned.mean(-1)
        rebinned = rebinned.mean(1)
    # Otherwise we are expanding
    else:
        if shape[0] % old_shape[0] != 0 or shape[1] % old_shape[1] != 0:
            raise ValueError("Your new shape should be a multiple of the original shape")
        # Get the size changes of the row and column separately
        row_sizer = shape[0] // old_shape[0]
        col_sizer = shape[1] // old_shape[1]
        # Check if we are expanding a single column (i.e. shape == (1,x))
        if shape[1] == 1:
            # Tile the range of col_sizer
            tiles = np.tile(np.array_split(np.arange(row_sizer), row_sizer),((shape[0] - 2) // row_sizer, shape[1]))
            # Get the differences between values
            differences = np.diff(a, axis=0) / row_sizer
            # Multiply differences array by tiles to get desired bins
            inferences = np.repeat(differences, row_sizer, axis=0) * tiles
            # Pad the inferences to get the same shape as above
            inferences = np.array_split(np.pad(inferences, (0, row_sizer))[:, 0], shape[0])
            # Add this to the original array that has been repeated to match the size of inference
            rebinned = inferences + np.repeat(a, row_sizer, axis=0)
        # Then we are expanding columns first and then rows (if there are rows to expand!)
        else:
            # Check if the original shape had one row, if ti does we need to change what axis we use
            if shape[0] == 1:
                # Use the rows (well row) first to get the correct columns
                ax = 0
            # If we have only one row, but we want to expand it to many rows and cols we need it in an array
            else:
                # This will ensure all resizing and padding works as expected
                if old_shape[0] == 1:
                    a = a.reshape((1,a.shape[0]))
                # Change the columns first
                ax = 1
            # Tile the range of col_sizer
            tiles = np.tile(np.arange(col_sizer), (old_shape[0], shape[1] // col_sizer-1))
            # Get the differences between values
            differences = np.diff(a, axis = ax) / col_sizer
            # Multiply differences array by tiles to get desired bins
            inferences = np.repeat(differences, col_sizer, axis = ax) * tiles
            # Pad the inferences to get the same shape as above
            inferences = np.pad(inferences, (0,col_sizer))[:-col_sizer]
            # Add this to the original array that has been repeated to match the size of inference
            col_rebinned = inferences + np.repeat(a, col_sizer, axis = ax)
            if col_rebinned.shape == shape:
                rebinned = col_rebinned
                if (shape[0] == 1):
                    rebinned = col_rebinned[0]
            else:
                if a.dtype == "int":
                    col_rebinned = np.fix(col_rebinned).astype("int")
                ax = 0
                # tile the range of row_sizer (but going down)
                tiles = np.tile(np.array_split(np.arange(row_sizer), row_sizer), ((shape[0]- row_sizer) // row_sizer, shape[1]))
                # Get the differences between the rows
                differences = np.diff(col_rebinned, axis = ax) / row_sizer
                # Multiply this by the tiles
                inferences = np.repeat(differences, row_sizer, axis = ax) * tiles
                # Pad the zeros for the last two rows, and remove the extra zeros to make inferences same shape as desired shape
                inferences = np.pad(inferences, (0,row_sizer))[:,:-row_sizer]
                # Now get our final array by adding the repeat of our columns rebinned to the inferences
                rebinned = inferences + np.repeat(col_rebinned, row_sizer, axis = ax)
    # In IDL the result is returned as an int if the original array was an int
    if a.dtype == "int":
            rebinned = np.fix(rebinned).astype("int")
    return rebinned