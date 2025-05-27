import cv2
import numpy as np
import os
from pathlib import Path

# ===== CONFIGURATION =====
# Path to the original image from the car
SOURCE_IMAGE = "./PerspectiveTransform/src/config/original.jpeg"

# Path to the bird's eye view (BEV) image
BEV_IMAGE = "./PerspectiveTransform/src/config/BEV.png"


# Directory containing all images to transform
INPUT_DIR = "/path/to/input/images"

# Directory to save transformed images
OUTPUT_DIR = "/path/to/output"

# ===== VALIDATION =====
# Validate paths
if not os.path.isfile(SOURCE_IMAGE):
    raise ValueError(f"Source image not found: {SOURCE_IMAGE}")
if not os.path.isfile(BEV_IMAGE):
    raise ValueError(f"BEV image not found: {BEV_IMAGE}")
if not os.path.isdir(INPUT_DIR):
    raise ValueError(f"Input directory not found: {INPUT_DIR}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PerspectiveTransformer:
    def __init__(self, output_dir):
        self.source_points = []
        self.dest_points = []
        self.current_point = 0
        self.point_names = ['Top Left', 'Bottom Left', 'Top Right', 'Bottom Right']
        self.source_image = None
        self.dest_image = None
        self.source_copy = None
        self.dest_copy = None
        self.current_window = "Source Image"
        self.transform_matrix = None
        self.output_dir = output_dir

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point < 4:
            if self.current_window == "Source Image":
                # Add point to source points
                self.source_points.append((x, y))
                
                # Draw point and label on source image
                cv2.circle(self.source_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.source_copy, self.point_names[self.current_point], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw lines between points
                if len(self.source_points) > 1:
                    cv2.line(self.source_copy, self.source_points[-2], self.source_points[-1], (0, 255, 0), 2)
                
                # Update source image display
                cv2.imshow("Source Image", self.source_copy)
                
                # Switch to destination image
                self.current_window = "Destination Image"
                print(f"Now select {self.point_names[self.current_point]} point in the bird's eye view")
                
            else:  # Destination Image
                # Add point to destination points
                self.dest_points.append((x, y))
                
                # Draw point and label on destination image
                cv2.circle(self.dest_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.dest_copy, self.point_names[self.current_point], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw lines between points
                if len(self.dest_points) > 1:
                    cv2.line(self.dest_copy, self.dest_points[-2], self.dest_points[-1], (0, 255, 0), 2)
                
                # Update destination image display
                cv2.imshow("Destination Image", self.dest_copy)
                
                # Move to next point
                self.current_point += 1
                
                # Switch back to source image if not done
                if self.current_point < 4:
                    self.current_window = "Source Image"
                    print(f"Select {self.point_names[self.current_point]} point in the original image")
                else:
                    self.compute_transform_matrix()

    def compute_transform_matrix(self):
        """Compute the perspective transformation matrix from the selected points"""
        # Convert points to numpy arrays
        pts1 = np.float32(self.source_points)
        pts2 = np.float32(self.dest_points)

        # Compute perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        print("Perspective Transformation Matrix:\n", self.transform_matrix)

    def transform_image(self, image):
        """Apply the perspective transform to an image"""
        if self.transform_matrix is None:
            raise ValueError("Transform matrix not computed. Please select points first.")
        
        # Get the size from the destination image
        output_size = (self.dest_image.shape[1], self.dest_image.shape[0])
        
        # Apply perspective transform
        transformed_image = cv2.warpPerspective(
            image,
            self.transform_matrix,
            output_size,
            flags=cv2.INTER_LINEAR
        )
        
        return transformed_image

    def process_images(self, source_path, dest_path, input_dir):
        """Process all images in the input directory"""
        # Load source and destination images
        self.source_image = cv2.imread(source_path)
        self.dest_image = cv2.imread(dest_path)
        
        if self.source_image is None:
            raise ValueError(f"Could not load source image from {source_path}")
        if self.dest_image is None:
            raise ValueError(f"Could not load destination image from {dest_path}")
        
        # Create copies for drawing
        self.source_copy = self.source_image.copy()
        self.dest_copy = self.dest_image.copy()
        
        # Create windows and set mouse callback
        cv2.namedWindow("Source Image")
        cv2.namedWindow("Destination Image")
        cv2.setMouseCallback("Source Image", self.mouse_callback)
        cv2.setMouseCallback("Destination Image", self.mouse_callback)
        
        # Display instructions
        print("Please select 4 points in this order:")
        print("1. Top Left")
        print("2. Bottom Left")
        print("3. Top Right")
        print("4. Bottom Right")
        print("You will select each point in both the original and bird's eye view images")
        print("Click 'q' to quit")
        
        # Show the images
        cv2.imshow("Source Image", self.source_copy)
        cv2.imshow("Destination Image", self.dest_copy)
        
        # Wait for point selection
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Process all images in the input directory
        input_path = Path(input_dir)
        for image_path in input_path.glob("*"):
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Could not load image: {image_path}")
                    continue
                
                # Apply transform
                transformed = self.transform_image(image)
                
                # Save transformed image
                output_path = Path(self.output_dir) / f"pt_{image_path.name}"
                cv2.imwrite(str(output_path), transformed)
                print(f"Saved transformed image to {output_path}")

def main():
    transformer = PerspectiveTransformer(OUTPUT_DIR)
    transformer.process_images(SOURCE_IMAGE, BEV_IMAGE, INPUT_DIR)

if __name__ == "__main__":
    main() 