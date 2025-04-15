# Perspective Transform

## About Perspective Transform
Perspective Transform maps points from one plane to another using a 3×3 homography matrix. This transformation can rotate, translate, and scale an image to mimic real-world perspective. In our project, perspective transform is used to perform a geometric transformation to simulate a top-down (bird’s-eye) view of the scene.

### **How Perspective Transform Works**
Follow the following steps to achieve perspective transform:

1. Identify the region of interest
2. Compute the homography matrix using anchor points or camera parameters and pose
3. Apply the transform to each pixel by multiplying the homography matrix


### **Two Approaches to Compute Homography**

1. **Anchor Points**
- Manually provide at least 4 point pairs (anchor -> target)
- If using OpenCV, use H = cv2.getPerspectiveTransform(anchor_pts, target_pts)
- [More on how to compute homography matrix using anchor points here](https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143)
2. **Pose & Instrinsics Method**
- Requires camera instrinsics(K), rotation(R), translation(t), and the ground plan normal vector(n)
- Compute homography matrix by using the following equation: H = K * (R - (t * n^T) / d) * K⁻¹


[Homography matrix and perspective transfrom procedure](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html?)

[How to select anchor points and target points for bird-eye view](https://nikolasent.github.io/opencv/2017/05/07/Bird%27s-Eye-View-Transformation.html?)


## Anchor Points

### What Are Anchor Points?

### How To Choose the Anchor Point

### How To Automate Selection of Anchor Point


## Best Practices for Perspective Transform In General

## Depth Camera in Perspective Transform

### Could We Use It for Our Need?

### What Data Is Needed from Depth Camera


## How to Access the Data Needed from Depth Camera

### Commonly Used Practices with Depth Camera


## Machine Learning in Perspective Transform