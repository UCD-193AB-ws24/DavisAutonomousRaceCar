import cv2
import numpy as np

'''
Click Order	Where to Click (in the mask image)

🔵 Top-Left       Left inner edge of the track where the curve starts
🟢 Bottom-Left    Left inner edge near the bottom of the image
🔴 Top-Right      Right outer edge of the track, same height as 🔵
🟡 Bottom-Right   Right outer edge near bottom-right of the image
'''

clicked_points = []

# Label colors for TL, BL, TR, BR
colors = [
    (255, 0, 0),    # 🔵 Blue for Top-Left
    (0, 255, 0),    # 🟢 Green for Bottom-Left
    (0, 0, 255),    # 🔴 Red for Top-Right
    (0, 255, 255)   # 🟡 Yellow for Bottom-Right
]

labels = ['TL', 'BL', 'TR', 'BR']

def click_event(event, x, y, flags, params):
    image, window_name = params
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        color = colors[len(clicked_points) - 1]
        label = labels[len(clicked_points) - 1]
        print(f"{label} → Point {len(clicked_points)}: ({x}, {y})")
        cv2.circle(image, (x, y), 6, color, -1)
        cv2.putText(image, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(window_name, image)

def main():
    img_path = "track1.png"
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ Error loading image: {img_path}")
        return

    image = cv2.resize(image, (640, 480))

    # Binarize
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Convert to color for visualization
    color_mask = cv2.cvtColor(binary_mask.copy(), cv2.COLOR_GRAY2BGR)
    window_name = "Click 4 Points"
    cv2.imshow(window_name, color_mask)
    cv2.setMouseCallback(window_name, click_event, (color_mask, window_name))

    print("👉 Click 4 points in this order: Top-Left 🔵, Bottom-Left 🟢, Top-Right 🔴, Bottom-Right 🟡")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        print("❌ You must click exactly 4 points.")
        return

    src_pts = np.float32(clicked_points)

    # Destination points
    dst_pts = np.float32([
        [150, 0],     # top-left
        [150, 480],   # bottom-left
        [490, 0],     # top-right
        [490, 480]    # bottom-right
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print("\n✅ Perspective Transform Matrix:\n", matrix)

    warped = cv2.warpPerspective(binary_mask, matrix, (640, 480), flags=cv2.INTER_NEAREST)

    # Display and save
    cv2.imshow("Top-Down View", warped)
    cv2.imwrite("top_down_output.png", warped)
    print("🖼️ Saved: top_down_output.png")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
