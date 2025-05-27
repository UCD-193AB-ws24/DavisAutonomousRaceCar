import cv2
import numpy as np
import os
from pathlib import Path

# ==== PATH CONFIGURATION ====
# Reference images for homography
CAR_IMAGE_PATH = "./PerspectiveTransform/src/config/original.png"
BEV_IMAGE_PATH = "./PerspectiveTransform/src/config/BEV.jpeg"

# Input and output directories
INPUT_DIR = "./PerspectiveTransform/src/inputs"  # Directory containing images to transform
OUTPUT_DIR = "./PerspectiveTransform/output/transformed_images"  # Directory to save transformed images

# Processing mode (0 for bitmask, 1 for regular images)
MODE = 1
# ==========================

def get_four_points(window_name, image):
    """Interactive function to get 4 points from an image"""
    clone = image.copy()
    points = []
    point_names = ['Top Left', 'Bottom Left', 'Top Right', 'Bottom Right']

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            # Draw point
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            # Draw label
            cv2.putText(clone, point_names[len(points)-1], 
                       (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw lines between points
            if len(points) > 1:
                cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nPlease click 4 points on the '{window_name}' window in this order:")
    print("1. Top Left - Select the top-left corner of the track")
    print("2. Bottom Left - Select the bottom-left corner of the track")
    print("3. Top Right - Select the top-right corner of the track")
    print("4. Bottom Right - Select the bottom-right corner of the track")
    print("Press 'q' to quit\n")

    while len(points) < 4:
        cv2.imshow(window_name, clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
            break

    cv2.destroyWindow(window_name)
    return points

def find_homography(car_path, bev_path):
    """Find homography matrix using car view and BEV images"""
    # Load images
    car_img = cv2.imread(car_path)
    bev_img = cv2.imread(bev_path)

    if car_img is None or bev_img is None:
        raise FileNotFoundError("One or both images not found. Check your paths.")

    # Resize both images to same size
    resize_width = min(car_img.shape[1], bev_img.shape[1])
    resize_height = min(car_img.shape[0], bev_img.shape[0])
    target_size = (resize_width, resize_height)

    car_resized = cv2.resize(car_img, target_size)
    bev_resized = cv2.resize(bev_img, target_size)

    print(f"Both images resized to: {target_size}")

    # Get points from both images
    car_points = get_four_points("1️⃣ Car View - Click Source Points", car_resized)
    bev_points = get_four_points("2️⃣ BEV View - Click Destination Points", bev_resized)

    # Compute homography
    src_pts = np.array(car_points, dtype=np.float32)
    dst_pts = np.array(bev_points, dtype=np.float32)
    H, status = cv2.findHomography(src_pts, dst_pts)

    print("\nComputed Homography Matrix:")
    print(H)

    # Apply the homography to the car view image
    warped_img = cv2.warpPerspective(car_resized, H, target_size)
    
    # Save the warped image
    output_path = Path("./PerspectiveTransform/output/warped_reference.png")
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), warped_img)
    print(f"\nSaved warped reference image to: {output_path}")

    return H, target_size

def apply_transform_to_directory(input_dir, output_dir, H, target_size, mode=1):
    """Apply the homography to all images in a directory"""
    input_path = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_count = 0
    success_count = 0

    print(f"\nProcessing images from: {input_dir}")
    print(f"Saving transformed images to: {output_dir}")
    print(f"Mode: {'Bitmask' if mode == 0 else 'Regular Images'}")

    for image_path in input_path.glob("*"):
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_count += 1
            print(f"\nProcessing {image_path.name}...")
            
            # Load image
            if mode == 0:  # Bitmask mode
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load bitmask: {image_path.name}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:  # Regular image mode
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Failed to load image: {image_path.name}")
                    continue
            
            # Resize image to match target size
            img_resized = cv2.resize(img, target_size)
            
            # Apply transformation
            warped_img = cv2.warpPerspective(img_resized, H, target_size)
            
            # Convert back to grayscale if in bitmask mode
            if mode == 0:
                warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            
            # Save transformed image
            output_path = Path(output_dir) / f"transformed_{image_path.name}"
            cv2.imwrite(str(output_path), warped_img)
            success_count += 1
            print(f"Saved transformed image to: {output_path}")

    print(f"\nProcessing complete!")
    print(f"Total images found: {image_count}")
    print(f"Successfully transformed: {success_count}")
    print(f"Failed to process: {image_count - success_count}")

def main():
    # Find homography
    print("\n=== Finding Homography ===")
    H, target_size = find_homography(CAR_IMAGE_PATH, BEV_IMAGE_PATH)

    # Apply transformation to directory
    print("\n=== Applying Transformation ===")
    apply_transform_to_directory(INPUT_DIR, OUTPUT_DIR, H, target_size, MODE)

if __name__ == "__main__":
    main() 