#include "test_run.cpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestRun>());
    rclcpp::shutdown();
    return 0;
}
