def compute_row_by_row_centerline(binary_mask):
    """
    Computes the centerline of a binary mask using a row-by-row boundary method.
    
    Parameters:
        binary_mask (np.ndarray): 2D numpy array (values 0 or 1).
        
    Returns:
        np.ndarray: An array of (x, y) points representing the centerline.
    """
    rows, cols = binary_mask.shape
    centerline_points = []

    for row in range(rows):
        # Find all column indices where the track is present in this row.
        indices = np.where(binary_mask[row, :] > 0)[0]
        if indices.size > 0:
            left = indices[0]
            right = indices[-1]
            mid = (left + right) / 2.0  # Compute the horizontal midpoint
            centerline_points.append((mid, row))
    
    return np.array(centerline_points)
