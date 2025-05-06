import cv2
import os
import numpy as np
from src.bitmask_filtering import (
    flood_fill_black,
    find_point_inside_track,
    convert_white_to_light_gray,
    convert_black_to_light_gray,
    resize_image,
    convert_border_gray_to_black
)

# === USER INPUT ===
INPUT_IMAGE_FILENAME = "track_1.png"  # Just the filename
OUTPUT_IMAGE_FILENAME = "track_1_220px.png"  # Just the filename
RESIZE_WIDTH = 220
ROTATION_ANGLE = 45  # Rotation angle in degrees (positive for counterclockwise)
# ===================

# Global Variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_PATH = os.path.join(ROOT_DIR, "src", "images", INPUT_IMAGE_FILENAME)
OUTPUT_IMAGE_PATH = os.path.join(ROOT_DIR, "outputs", OUTPUT_IMAGE_FILENAME)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)

def rotate_image(image, angle):
    """
    Rotate image by specified angle in degrees and crop to remove blank space.
    
    Args:
        image: Input image (numpy array)
        angle: Rotation angle in degrees (positive for counterclockwise)
        
    Returns:
        numpy array: Rotated and cropped image
    """
    print(f"Rotating image by {angle} degrees...")
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center = (width // 2, height // 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to fit the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    # Compute new dimensions
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform the rotation with nearest neighbor interpolation to prevent color blending
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                            flags=cv2.INTER_NEAREST, 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=[0, 0, 0])  # Fill new areas with black
    
    # Find non-black pixels to determine crop boundaries
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Crop the image
        rotated = rotated[y:y+h, x:x+w]
    
    return rotated

def main():
    try:
        # Load the input image
        if not os.path.exists(INPUT_IMAGE_PATH):
            raise FileNotFoundError(f"Input image not found at: {INPUT_IMAGE_PATH}")
            
        image = cv2.imread(INPUT_IMAGE_PATH)
        if image is None:
            raise ValueError(f"Could not read image at: {INPUT_IMAGE_PATH}")
        
        # Step 1: Find a point inside the track
        print("\nStep 1: Finding point inside track...")
        inner_point = find_point_inside_track(image)
        if inner_point is None:
            print("Could not find a suitable point inside track")
            return
            
        start_x, start_y = inner_point
        
        # Step 2: Convert white pixels to light gray
        print("\nStep 2: Converting white pixels to light gray...")
        gray_image = convert_white_to_light_gray(image)
        
        # Step 3: Apply flood fill
        print("\nStep 3: Applying flood fill...")
        filled_image = flood_fill_black(gray_image, start_x, start_y)
        
        # Step 4: Convert black pixels to light gray
        print("\nStep 4: Converting black pixels to light gray...")
        final_image = convert_black_to_light_gray(filled_image)
        
        # Step 5: Resize and rotate the final image
        print("\nStep 5: Resizing and rotating final image...")
        resized_final = resize_image(final_image, RESIZE_WIDTH)
        if ROTATION_ANGLE != 0:
            resized_final = rotate_image(resized_final, ROTATION_ANGLE)
        
        # Step 6: Convert border gray pixels to black
        print("\nStep 6: Converting border gray pixels to black...")
        final_border = convert_border_gray_to_black(resized_final)
        
        # Save the final result
        output_filled_path = OUTPUT_IMAGE_PATH.replace(".png", f"_filled_{ROTATION_ANGLE}deg.png")
        cv2.imwrite(output_filled_path, final_border)
        print(f"\nFinal image saved to: {output_filled_path}")
        
        # Show the final result
        cv2.imshow("Final Result", final_border)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main() 