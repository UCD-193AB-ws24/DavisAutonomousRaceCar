# Davis Autonomous Race Car

This repository contains the codebase for the UC Davis Autonomous Race Car project, built on the F1TENTH platform. The system enables autonomous racing capabilities through a combination of localization, mapping, and control algorithms.

## Repository Structure

### Core Components
- `darc_f1tenth_system/`: Main ROS2 package containing the core autonomous driving system
- `particle_filter/`: Implementation of particle filter for robot localization
- `bitmask_filtering/`: Algorithms for processing and filtering sensor data
- `automatic_bitmask_stitching/`: Tools for creating and managing environment maps
- `f1tenth_gym_ros/`: ROS2 integration with the F1TENTH gym simulator
- `tests/`: Unit and integration tests for the system
- `docs/`: Additional documentation and guides

### Supporting Tools
- `PerspectiveTransform/`: Tools for camera perspective transformation
- `Raceline-Optimization/`: Algorithms for optimal racing line computation
- `Stitching/`: Map stitching and processing utilities

## System Architecture

The autonomous race car system consists of several key components:

1. **Localization System**
   - Particle filter-based localization
   - Map-based position estimation
   - Real-time pose tracking

2. **Perception System**
   - LiDAR data processing
   - Bitmask filtering for obstacle detection
   - Environment mapping

3. **Control System**
   - Pure pursuit path following
   - Velocity and steering control
   - Obstacle avoidance

4. **Planning System**
   - Raceline optimization
   - Path planning
   - Trajectory generation

## Getting Started

# **Autonomous Race Car Setup Guide (how to drive autonomously given that we we provide it a fully functioning map; PNG and YAML)**

## **Step 1: Verify Map Placement**
1. **Ensure the map files are in the correct directory:**  
   - Place your YAML and PNG files in `f1tenth_stack/maps/`.

2. **Check file consistency:**  
   - Open the YAML file and confirm:
     - It correctly references the PNG file.
     - Parameters such as `resolution`, `origin`, and `occupied_thresh` are set properly.

---

## **Step 2: Update Configuration Files**
Now that your map is in place, update the configuration files so the system knows where to find it.

### **1. Update the Particle Filter Configuration**
- Open `particle_filter.yaml` (found in `f1tenth_stack/config/`).
- Set the `map_yaml_path` to point to your YAML file:
  ```yaml
  map_yaml_path: /path/to/f1tenth_stack/maps/your_map.yaml
  ```

### **2. Update the Raceline (if applicable)**
If you have a precomputed raceline CSV:
- Open `pure_pursuit.yaml` and update the `trajectory` parameter:
  ```yaml
  trajectory: /path/to/f1tenth_stack/racelines/your_raceline.csv
  ```
- Open `obs_detect.yaml` and update `spline_file_name` similarly:
  ```yaml
  spline_file_name: /path/to/f1tenth_stack/racelines/your_raceline.csv
  ```

---

## **Step 3: Build and Source Your ROS2 Workspace**
### **1. Navigate to Your Workspace**
```bash
cd ~/f1tenth_ws
```

### **2. Build the Workspace**
```bash
colcon build
```
This compiles your ROS2 packages with the updated configurations.

### **3. Source the Setup Files**
```bash
source /opt/ros/foxy/setup.bash
source ~/f1tenth_ws/install/setup.bash
```
This ensures your system recognizes the newly built packages.

### **4. (Optional) Automate Sourcing**
To avoid manually sourcing each time, add it to your `.bashrc`:
```bash
echo 'source /opt/ros/foxy/setup.bash && source ~/f1tenth_ws/install/setup.bash' >> ~/.bashrc
```

---

## **Step 4: Start the Car's ROS2 System**
Now, launch the core ROS2 system that manages hardware components.

### **1. Start the Base System**
```bash
ros2 launch f1tenth_stack bringup_launch.py
```
This will:
- Start the **VESC driver** (for motor speed and steering).
- Start the **LiDAR driver** (for scanning the environment).
- Initialize other necessary hardware nodes.

### **2. Verify Everything is Running**
- Check for errors in the terminal.
- List all available ROS2 topics:
  ```bash
  ros2 topic list
  ```
  You should see topics like:
  ```
  /scan   # LiDAR data
  /vesc/odom   # Odometry data
  ```

---

## **Step 5: Start Localization (Using the Particle Filter)**
Now, the car needs to **localize itself** on the map.

### **1. Launch the Particle Filter**
```bash
ros2 launch f1tenth_stack particle_filter_launch.py
```
This will:
- Start **particle filter localization**.
- Load the **pre-existing map**.
- Begin estimating the car’s real-time position.

### **2. Set the Initial Pose in RViz**
1. Open **RViz** (it should launch automatically).
2. Click **"2D Pose Estimate"** from the toolbar.
3. Click and drag on the map where the car is physically located.

Now, the car’s estimated pose should be visualized as a cluster of particles.

### **3. Verify Localization**
Run:
```bash
ros2 topic echo /particle_filter/pose
```
You should see real-time estimated pose values.

---

## **Step 6: Start Autonomous Driving 🚗**
Now that localization is working and the car knows where it is on the map, start the **autonomous driving system**.

### **1. Launch the Autonomous Control Stack**
Run:
```bash
ros2 launch f1tenth_stack autonomous_launch.py
```
This will:
- Load the **Pure Pursuit** algorithm (for following the precomputed raceline).
- Activate **Obstacle Detection** (if enabled).
- Start publishing drive commands (`/cmd_vel` or Ackermann messages).

### **2. Enable Autonomous Mode**
If using a joystick:
- **Hold the RB button** to enable autonomous driving.

If using ROS2 commands:
- You may need to manually publish an enable signal.

---

## **Step 7: Verify Movement**
Check if the car moves along the raceline.

### **1. Check if Drive Commands Are Being Sent**
```bash
ros2 topic echo /cmd_vel
```
- If commands are appearing but the car isn't moving, check the **VESC motor controller connection**.

### **2. Manually Send a Movement Command**
Test by sending a forward motion command:
```bash
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```
This should briefly move the car forward.

---

## **Step 8: Troubleshooting (If the Car Does Not Move)**
### **1. Check if `/cmd_vel` is being published**
```bash
ros2 topic list | grep cmd_vel
```
If the topic is missing, the autonomous node is not sending commands.

### **2. Check if the motor control node is running**
```bash
ros2 node list
```
If the VESC driver is missing, restart `bringup_launch.py`.

### **3. Restart the System**
If nothing else works:
```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch f1tenth_stack particle_filter_launch.py
ros2 launch f1tenth_stack autonomous_launch.py
```
This resets all nodes and often fixes connectivity issues.

---

## **Final Confirmation**
Once the car moves and follows the raceline, **autonomous mode is fully active!** 🎉
