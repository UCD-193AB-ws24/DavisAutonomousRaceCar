import cv2
from perspective_transform import apply_perspective_transform
from thinning_algorithm import zhang_suen_thinning

def main():
    # Load and apply perspective transform
    image_path = "test3_original.png"  # Update with your correct path
    mask_path = "test3.png"  # Update with your correct path
    transformed_image, transformed_mask = apply_perspective_transform(image_path, mask_path)
    
    # Apply Zhang-Suen Thinning to the transformed mask
    skeletonized_mask = zhang_suen_thinning(transformed_mask)
    
    # Display Results
    cv2.imshow("Transformed Image", transformed_image)
    cv2.imshow("Skeletonized Mask (Thinned)", skeletonized_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
