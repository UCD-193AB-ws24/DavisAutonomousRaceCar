import cv2

def select_point(event, x, y, flags, param):
    """ Callback function to capture the coordinates of a clicked point. """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selected Point: ({x}, {y})")
        param.append((x, y))  # Store the point in the list

def main():
    # Load the image
    image_path = "after.png"  # Change this to your image file
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return
    
    # Resize the image to 800x600
    resized_image = cv2.resize(image, (800, 600))
    
    # Create a window and set the mouse callback function
    cv2.namedWindow("Select a Point")
    selected_points = []
    cv2.setMouseCallback("Select a Point", select_point, selected_points)
    
    while True:
        cv2.imshow("Select a Point", resized_image)
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to exit
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Print all selected points
    print("Final Selected Points:", selected_points)

if __name__ == "__main__":
    main()
