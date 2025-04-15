#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

class TestRun : public rclcpp::Node {
public:
    TestRun() : Node("test_run_node") {
        // Declare drive topic parameter
        this->declare_parameter("drive_topic", "/drive");

        // Get parameter value
        drive_topic_ = this->get_parameter("drive_topic").as_string();

        // Create publisher for drive commands
        drive_publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);

        // Subscribe to elapsed time
        time_subscriber_ = this->create_subscription<std_msgs::msg::Float64>(
            "elapsed_time", 10, std::bind(&TestRun::time_callback, this, std::placeholders::_1));
    }

private:
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr time_subscriber_;
    std::string drive_topic_;

    void time_callback(const std_msgs::msg::Float64::SharedPtr msg) {
        double elapsed_time = msg->data;
        RCLCPP_INFO(this->get_logger(), "Elapsed Time Received: %.2f seconds", elapsed_time);

        // Call functions based on elapsed time
        if (elapsed_time < 3.0) {
            go_straight();
        } else if (elapsed_time >= 3.0 && elapsed_time < 5.0) {
            turn_right();
        } else if (elapsed_time >= 5.0 && elapsed_time < 7.0) {
            turn_left();
        } else if (elapsed_time >= 7.0 && elapsed_time < 9.0) {
            reverse();
        } else {
            stop();
        }
    }

    void go_straight() {
        publish_drive_message(2.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Going Straight");
    }

    void turn_right() {
        publish_drive_message(1.5, -0.5);
        RCLCPP_INFO(this->get_logger(), "Turning Right");
    }

    void turn_left() {
        publish_drive_message(1.5, 0.5);
        RCLCPP_INFO(this->get_logger(), "Turning Left");
    }

    void reverse() {
        publish_drive_message(-1.5, 0.0);
        RCLCPP_INFO(this->get_logger(), "Reversing");
    }

    void stop() {
        publish_drive_message(0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Stopping");
    }

    void publish_drive_message(double speed, double steering_angle) {
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.speed = speed;
        drive_msg.drive.steering_angle = steering_angle;
        drive_publisher_->publish(drive_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestRun>());
    rclcpp::shutdown();
    return 0;
}
