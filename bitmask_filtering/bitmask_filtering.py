import cv2
import os
from src.bitmask_filtering import (
    flood_fill_black,
    find_point_inside_track,
    convert_white_to_light_gray,
    convert_black_to_light_gray,
    resize_image,
    convert_border_gray_to_black
)

# === USER INPUT ===
INPUT_IMAGE_FILENAME = "bitmask_circle.png"  # Just the filename
OUTPUT_IMAGE_FILENAME = "RaceTrack_Circle_220px.png"  # Just the filename
RESIZE_WIDTH = 220
# ===================

# Global Variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_PATH = os.path.join(ROOT_DIR, "src", "images", INPUT_IMAGE_FILENAME)
OUTPUT_IMAGE_PATH = os.path.join(ROOT_DIR, "outputs", OUTPUT_IMAGE_FILENAME)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)

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
        
        # Step 5: Resize the final image
        print("\nStep 5: Resizing final image...")
        resized_final = resize_image(final_image, RESIZE_WIDTH)
        
        # Step 6: Convert border gray pixels to black
        print("\nStep 6: Converting border gray pixels to black...")
        final_border = convert_border_gray_to_black(resized_final)
        
        # Save the final result
        output_filled_path = OUTPUT_IMAGE_PATH.replace(".png", "_filled.png")
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