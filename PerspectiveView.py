import cv2
import numpy as np

def get_sam_mask(frame):
    """
    Placeholder for SAM segmentation.
    In your actual implementation, replace this with a call to SAM
    that returns a binary mask (e.g., track pixels=255, background=0).
    For demonstration, we convert the image to grayscale and threshold it.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # This threshold is arbitrary. SAM's output would be your binary segmentation mask.
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return mask

# Open your video file
vidcap = cv2.VideoCapture("myDemoVideo.mp4")
success, image = vidcap.read()

while success:
    success, image = vidcap.read()
    if not success:
        break
    
    # Resize the frame for consistency
    frame = cv2.resize(image, (640, 480))
    
    # Get segmentation mask from SAM (simulated here)
    sam_mask = get_sam_mask(frame)
    
    # --- Perspective Transformation Parameters ---
    # Define source points (from your original image)
    tl = (222, 387)  # top-left
    bl = (70, 472)   # bottom-left
    tr = (400, 380)  # top-right
    br = (538, 472)  # bottom-right

    # (Optional) Draw these points on the original frame for visualization
    for point in [tl, bl, tr, br]:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Define destination points for a bird’s-eye view
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Compute the transformation matrix and apply it
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Apply perspective transform to the SAM mask and the original frame
    transformed_mask = cv2.warpPerspective(sam_mask, matrix, (640, 480))
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Display the results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("SAM Mask", sam_mask)
    cv2.imshow("Transformed Mask", transformed_mask)
    cv2.imshow("Transformed Frame", transformed_frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

vidcap.release()
cv2.destroyAllWindows()
