#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include <chrono>

using namespace std::chrono_literals;

class ElapsedTimePublisher : public rclcpp::Node {
public:
    ElapsedTimePublisher() : Node("elapsed_time_publisher") {
        // Create publisher for elapsed time
        publisher_ = this->create_publisher<std_msgs::msg::Float64>("elapsed_time", 10);

        // Timer to publish every 0.1 seconds
        timer_ = this->create_wall_timer(100ms, std::bind(&ElapsedTimePublisher::timer_callback, this));

        // Record the start time
        start_time_ = this->now();
    }

private:
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time start_time_;

    void timer_callback() {
        // Compute elapsed time
        double elapsed_seconds = (this->now() - start_time_).seconds();

        // Create message
        std_msgs::msg::Float64 msg;
        msg.data = elapsed_seconds;

        // Publish elapsed time
        publisher_->publish(msg);

        // Log output
        RCLCPP_INFO(this->get_logger(), "Elapsed Time: %.2f seconds", elapsed_seconds);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ElapsedTimePublisher>());
    rclcpp::shutdown();
    return 0;
}
