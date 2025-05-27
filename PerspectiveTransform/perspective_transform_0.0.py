import cv2
import numpy as np

# ==== USER INPUT WITH DEFAULTS ====
default_car_path = "./PerspectiveTransform/src/config/original.png"
default_bev_path = "./PerspectiveTransform/src/config/BEV.jpeg"
#target_input = "./PerspectiveTransform/src/inputs/frame_0123_1745869128166226650.png"
target_input = "./PerspectiveTransform/src/config/original.png"

car_input = input(f"Enter car-view image path [default: {default_car_path}]: ").strip()
bev_input = input(f"Enter BEV image path [default: {default_bev_path}]: ").strip()


CAR_IMAGE_PATH = car_input if car_input else default_car_path
BEV_IMAGE_PATH = bev_input if bev_input else default_bev_path
# ==================================

# === Load and check images ===
car_img = cv2.imread(CAR_IMAGE_PATH)
bev_img = cv2.imread(BEV_IMAGE_PATH)
tar_img = cv2.imread(target_input)

if car_img is None or bev_img is None:
    raise FileNotFoundError("One or both images not found. Check your paths.")

# === Resize both images to same size ===
resize_width = min(car_img.shape[1], bev_img.shape[1])
resize_height = min(car_img.shape[0], bev_img.shape[0])
target_size = (resize_width, resize_height)

car_resized = cv2.resize(car_img, target_size)
bev_resized = cv2.resize(bev_img, target_size)
tar_resized = cv2.resize(tar_img, target_size)

print(f"Both images resized to: {target_size}")

# === Point collection logic ===
def get_four_points(window_name, image):
    clone = image.copy()
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(clone, f"{len(points)}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"Please click 4 points on the '{window_name}' window.")
    while len(points) < 4:
        cv2.imshow(window_name, clone)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to cancel
            break

    cv2.destroyWindow(window_name)
    return points

# === Step 1: Select source points from CAR image ===
car_points = get_four_points("1️⃣ Car View (Resized) - Click Source Points", car_resized)

# === Step 2: Select destination points from BEV image ===
bev_points = get_four_points("2️⃣ BEV View (Resized) - Click Destination Points", bev_resized)

# === Compute homography and warp ===
src_pts = np.array(car_points, dtype=np.float32)  # FROM
dst_pts = np.array(bev_points, dtype=np.float32)  # TO

H, status = cv2.findHomography(src_pts, dst_pts)
print("\nComputed Homography Matrix:")
print(H)

warped_img = cv2.warpPerspective(tar_resized, H, target_size)

# === Display and save result ===
cv2.imshow("Warped Image", warped_img)
cv2.imwrite("output_warped.png", warped_img)
print("Warped image saved as output_warped.png")

cv2.waitKey(0)
cv2.destroyAllWindows()
