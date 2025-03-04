import cv2
import numpy as np

def get_perspective_transform(src_points, dst_points):
    """
    Computes the perspective transform matrix and its inverse.
    """
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def warp_image(image, M, width, height):
    """
    Applies a perspective warp to the image using the matrix M.
    """
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped

def calculate_midpoint_from_mask(mask):
    """
    Calculates the midpoint of the track boundaries from a binary mask.
    Assumes the largest contour represents the track.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Assume the largest contour corresponds to the track boundary
    contour = max(contours, key=cv2.contourArea)
    
    # Get the leftmost and rightmost points in the contour
    left = tuple(contour[contour[:, :, 0].argmin()][0])
    right = tuple(contour[contour[:, :, 0].argmax()][0])
    
    # The midpoint is the average of the leftmost and rightmost points
    midpoint = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
    return midpoint

def main():
    # Open a video stream or video file
    cap = cv2.VideoCapture(0)  # Change to video file if needed

    # Define source (from camera) and destination (bird's-eye view) points.
    # These must be calibrated for your specific setup.
    src_points = [[100, 200], [220, 200], [20, 0], [300, 0]]
    dst_points = [[100, 240], [220, 240], [100, 0], [220, 0]]
    width, height = 320, 240

    # Compute perspective transform matrices
    M, M_inv = get_perspective_transform(src_points, dst_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply perspective warp to obtain bird's-eye view
        warped = warp_image(frame, M, width, height)
        
        # --- Segmentation ---
        # For demonstration, we use a simple threshold as a placeholder.
        # Replace the following lines with your actual segmentation mask when available.
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Calculate the midpoint from the binary mask
        midpoint = calculate_midpoint_from_mask(mask)
        
        # Visualize results by drawing the midpoint on the warped image
        display = warped.copy()
        if midpoint is not None:
            cv2.circle(display, midpoint, 5, (0, 0, 255), -1)
            cv2.putText(display, f"Midpoint: {midpoint}", (midpoint[0]-40, midpoint[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show windows for debugging/visualization
        cv2.imshow("Warped Image", warped)
        cv2.imshow("Mask", mask)
        cv2.imshow("Midpoint Display", display)
        
        # Exit on pressing 'Esc'
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
