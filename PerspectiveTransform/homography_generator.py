import numpy as np
import cv2
import os

# ============================
# 🧾 User Configuration
# ============================

# Path to the input image
image_Path = "your_image.png"  # Replace with your actual image path

# Camera intrinsics (from calibration)
# Subscribe the YAML for configuration
fx = 
fy = 

# Camera height from ground in meters
# This is always fixed
camera_height = 0.0

# Orientation from IMU (degrees)
# This can be subscribed from camera's IMU values
pitch_deg =   # Up/down tilt
yaw_deg =       # Left/right turn
roll_deg =     # Side tilt

# Ground is assumed flat, with normal pointing up in world frame
ground_normal = np.array([0, 1, 0])

# ============================
# 🚀 Processing
# ============================

# Load image to get resolution
if not os.path.exists(image_Path):
    raise FileNotFoundError(f"Image not found at path: {image_Path}")
image = cv2.imread(image_Path)
if image is None:
    raise ValueError(f"Could not read image from: {image_Path}")
image_height, image_width = image.shape[:2]

# Principal point
cx = image_width / 2
cy = image_height / 2

# Intrinsic matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

# Convert IMU angles to radians
pitch = np.deg2rad(pitch_deg)
yaw = np.deg2rad(yaw_deg)
roll = np.deg2rad(roll_deg)

# Rotation matrices
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(pitch), -np.sin(pitch)],
    [0, np.sin(pitch),  np.cos(pitch)]
])

Ry = np.array([
    [np.cos(yaw), 0, np.sin(yaw)],
    [0, 1, 0],
    [-np.sin(yaw), 0, np.cos(yaw)]
])

Rz = np.array([
    [np.cos(roll), -np.sin(roll), 0],
    [np.sin(roll),  np.cos(roll), 0],
    [0, 0, 1]
])

# Combined rotation matrix
R = Rz @ Ry @ Rx

# Translation vector
T = np.array([[0], [camera_height], [0]])

# Plane normal and distance to ground
n = ground_normal.reshape(3, 1)
d = camera_height

# Compute homography matrix
H_matrix = R - (T @ n.T) / d
H = K @ H_matrix @ np.linalg.inv(K)

# ============================
# 🖨️ Output
# ============================

print(f"✅ Image resolution: {image_width}x{image_height}")
print("📸 Intrinsic Matrix (K):\n", K)
print(f"🧭 Provided Orientation (deg): Pitch={pitch_deg}, Yaw={yaw_deg}, Roll={roll_deg}")
print("🔁 Homography Matrix (H):\n", H)
