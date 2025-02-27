import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize

def compute_medial_axis(binary_mask):
    """
    Computes the medial axis (skeleton) and distance transform of the binary mask.
    
    Parameters:
        binary_mask (np.ndarray): 2D numpy array (values 0 or 1).
    
    Returns:
        skel (np.ndarray): Binary image with the medial axis.
        distance (np.ndarray): Distance transform (each pixel's distance to the nearest boundary).
    """
    skel, distance = medial_axis(binary_mask, return_distance=True)
    return skel, distance

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

def main():
    # Create a synthetic binary mask representing a track.
    # For a more realistic scenario, replace this with your segmentation output.
    height, width = 200, 400
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw a track-like rectangle in the middle and add a slight curvature.
    for i in range(80, 120):
        shift = int(20 * np.sin((i - 80) / 40 * np.pi))
        mask[i, 100 + shift:300 + shift] = 1

    binary_mask = mask

    # Compute the medial axis using scikit-image.
    skel, distance = compute_medial_axis(binary_mask)
    
    # Compute the centerline using the row-by-row method.
    row_centerline = compute_row_by_row_centerline(binary_mask)
    
    # Plot the results.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(binary_mask, cmap='gray')
    axes[0].set_title("Original Binary Mask")
    
    axes[1].imshow(skel, cmap='gray')
    axes[1].set_title("Medial Axis (Skeletonization)")
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].scatter(row_centerline[:, 0], row_centerline[:, 1], color='red', s=10)
    axes[2].set_title("Row-by-Row Centerline")
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
