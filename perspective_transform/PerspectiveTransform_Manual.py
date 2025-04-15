import cv2
import numpy as np

def main():
    # Load bitmask image (grayscale)
    img_path = "track1.png"
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: could not read {img_path}")
        return
    # Print original image size
    print(f"Original image dimensions: Width={mask.shape[1]}px, Height={mask.shape[0]}px")

    # Resize to standard size (optional)
    original_size = (640, 480)
    mask = cv2.resize(mask, original_size)
    # Print image size
    print(f"Image dimensions: Width={original_size[0]}px, Height={original_size[1]}px")
    
    # Binarize the mask just to be sure
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # Define source and destination points
    # -----------------------------
    src_pts = np.float32([
        [200, 250],   # top-left
        [50, 470],    # bottom-left
        [440, 250],   # top-right
        [610, 470]    # bottom-right
    ])

    dst_pts = np.float32([
        [150, 0],     # top-left
        [150, 480],   # bottom-left
        [490, 0],     # top-right
        [490, 480]    # bottom-right
    ])

    # -----------------------------
    # Show source points on image
    # -----------------------------
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow
    labels = ['TL', 'BL', 'TR', 'BR']

    for i, (pt, color) in enumerate(zip(src_pts, colors)):
        pt = tuple(int(v) for v in pt)
        cv2.circle(color_mask, pt, 6, color, -1)
        cv2.putText(color_mask, labels[i], (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Source Points Overlay", color_mask)
    cv2.imwrite("source_points_overlay.png", color_mask)

    # -----------------------------
    # Perspective transform
    # -----------------------------
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print("Perspective Transform Matrix:\n", matrix)

    warped = cv2.warpPerspective(binary_mask, matrix, (640, 480), flags=cv2.INTER_NEAREST)

    # -----------------------------
    # Show result
    # -----------------------------
    cv2.imshow("Original Mask", binary_mask)
    cv2.imshow("Top-Down View", warped)
    # Save the top-down view
    cv2.imwrite("top_down_view.png", warped)
    print("🖼️ Saved: top_down_view.png")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
