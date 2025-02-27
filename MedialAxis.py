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
