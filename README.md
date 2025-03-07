# AutonomousRaceCar

Step 1: Verify Map Placement
Location:
Ensure that your map files (the YAML file and the corresponding PNG image) are in the correct directory (typically something like f1tenth_stack/maps).
File Consistency:
Open the YAML file to confirm it correctly references the PNG file and that parameters (resolution, origin, etc.) are set appropriately.

Step 2: Update Configuration Files
Now that your map files are in place, the next step is to ensure your system's configuration files reference them correctly.

Particle Filter Configuration:
Open the particle_filter.yaml file (usually found in the f1tenth_stack/config directory).

Update the map_yaml_path parameter so that it points to your map’s YAML file. For example:
bash
Copy
Edit
map_yaml_path: /path/to/f1tenth_stack/maps/your_map.yaml
Raceline (if applicable):
If you have a pre-computed raceline CSV:

Open pure_pursuit.yaml and update the trajectory parameter to reference your raceline CSV file.
Similarly, check obs_detect.yaml for any parameters like spline_file_name that should point to your raceline.
Once you've updated these configuration files, your system will know where to load the map (and raceline, if using one) for localization and path planning.

Step 3: Build and Source Your ROS2 Workspace
Now that your configuration files are updated, the next step is to build your ROS2 workspace and ensure your environment is correctly sourced. This makes sure all your nodes and settings are available for launch.

Navigate to Your Workspace:
Open a terminal and go to your ROS2 workspace directory (e.g., ~/f1tenth_ws).

Build the Workspace:
Run the build command:

colcon build

This compiles your packages and integrates your updated configurations.

Source the Setup Files:
Once the build completes, source the ROS2 setup files:

source /opt/ros/foxy/setup.bash

source ~/f1tenth_ws/install/setup.bash

This ensures your terminal session is aware of the newly built packages and configurations.

(Optional) Add to Bashrc:
To avoid sourcing manually every time, add the source commands to your ~/.bashrc file:

echo 'source /opt/ros/foxy/setup.bash && source ~/f1tenth_ws/install/setup.bash' >> ~/.bashrc

Once you've built and sourced your workspace, your system is ready to launch the nodes that use your map for localization and path planning.

Step 4: Start the Car's ROS2 System
Now that your workspace is built and sourced, the next step is to start the core ROS2 system, which includes necessary hardware drivers and sensor integration. This ensures that the car’s components (LiDAR, motor controller, etc.) are running before you attempt localization and autonomous driving.

1. Launch the Base System
Run the following command to bring up the core system:

ros2 launch f1tenth_stack bringup_launch.py

This launch file typically starts:

- The VESC driver (which controls motor speed and steering).
- The LiDAR driver (for scanning the environment).
- Other necessary hardware-related nodes.

2. Verify Everything is Running
Check for Errors: If the launch output shows errors related to missing hardware (e.g., "VESC not found"), ensure that all hardware is connected properly and powered on.

View Available ROS2 Topics: Run:

ros2 topic list

This should display topics like /scan (LiDAR data) and /vesc/odom (odometry).
If everything is running correctly and the topics are visible, the car’s system is ready for localization.

Step 5: Start Localization (Using the Particle Filter)
Now that the base system is running, the car needs to localize itself on the map before it can drive autonomously. This is done using the particle filter, which estimates the car’s position by matching sensor data (LiDAR scans) to the pre-existing map.

1. Launch the Particle Filter for Localization
Run:

ros2 launch f1tenth_stack particle_filter_launch.py

This will:

Start the particle filter localization algorithm.
Load the pre-existing map (YAML + PNG) that you configured.
Begin estimating the car’s position in real-time.

2. Set the Initial Pose in RViz
Since the particle filter needs an initial guess of where the car is, you must manually set the car’s initial pose in RViz:

Open RViz (it should launch automatically).
Click "2D Pose Estimate" in the RViz toolbar.
Click and drag on the map to place the car approximately where it is in reality.

The car’s estimated pose should now be visualized as a cluster of particles around this location.

3. Verify Localization
The car’s estimated position should align with its real-world location.
The particle cloud should converge as the car moves slightly.

Run:

ros2 topic echo /particle_filter/pose

This should show real-time estimated pose values.

Step 6: Start Autonomous Driving 🚗
Now that localization is working and the car knows where it is on the map, the next step is to start the autonomous driving system, which will command the car to follow the raceline and navigate autonomously.

1. Launch the Autonomous Control Stack
Run the following command to start the autonomous driving algorithms (Pure Pursuit, Gap Follow, and Obstacle Detection):

ros2 launch f1tenth_stack autonomous_launch.py

This will:

Load the Pure Pursuit algorithm (for following the precomputed raceline).
Activate Obstacle Detection (if enabled).
Start publishing drive commands (/cmd_vel or Ackermann messages) to move the car.

2. Enable Autonomous Mode
If your system is set up to use a joystick as an emergency override, you may need to manually enable autonomous driving:

Hold the RB button (on the joystick) to allow the car to start following the raceline.
If using a keyboard or ROS2 command, you may need to manually publish an enable signal.

3. Verify Movement
Check if the car moves forward along the raceline.
If the car does not move:
Check the control commands:

ros2 topic echo /cmd_vel

If this shows velocity commands but the car isn’t moving, check the motor controller (VESC) connection.
Manually send a command to test movement:

ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

This should make the car move forward briefly.

4. Troubleshooting (If the Car Does Not Move)
Check if /cmd_vel is being published.

ros2 topic list | grep cmd_vel

If you don’t see /cmd_vel, it may not be publishing commands.

Check motor controller logs.
Run:

ros2 node list

Ensure the motor control node (e.g., VESC) is running. If not, restart bringup_launch.py.






