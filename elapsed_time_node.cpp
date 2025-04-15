#include "elapsed_time_publisher.cpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ElapsedTimePublisher>());
    rclcpp::shutdown();
    return 0;
}
