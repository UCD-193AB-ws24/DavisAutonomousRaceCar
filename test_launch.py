from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # Elapsed Time Publisher Node
    elapsed_time_node = Node(
        package="test_run",
        executable="elapsed_time_publisher",
        name="elapsed_time_publisher",
        output="screen"
    )

    # Test Run Node (Subscribes to elapsed time)
    test_run_node = Node(
        package="test_run",
        executable="test_run",
        name="test_run",
        output="screen",
        parameters=[{"drive_topic": "/drive"}]
    )

    # Add Nodes to Launch Description
    ld.add_action(elapsed_time_node)
    ld.add_action(test_run_node)

    return ld
