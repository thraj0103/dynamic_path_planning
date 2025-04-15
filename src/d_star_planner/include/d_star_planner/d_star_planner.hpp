#ifndef D_STAR_PLANNER__D_STAR_PLANNER_HPP_
#define D_STAR_PLANNER__D_STAR_PLANNER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "tf2_ros/buffer.h"

namespace d_star_planner
{

struct Cell
{
  int x, y;
  bool operator==(const Cell & other) const { return x == other.x && y == other.y; }
};

class DStarPlanner : public nav2_core::GlobalPlanner
{
public:
  DStarPlanner();
  ~DStarPlanner();

  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void activate() override;
  void deactivate() override;
  void cleanup() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

protected:
  bool makePlan(
    const geometry_msgs::msg::Pose & start,
    const geometry_msgs::msg::Pose & goal,
    double tolerance,
    nav_msgs::msg::Path & plan);

  bool worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my);
  void mapToWorld(double mx, double my, double & wx, double & wy);
  void clearRobotCell(unsigned int mx, unsigned int my);
  bool isPlannerOutOfDate();

  void smoothApproachToGoal(
    const geometry_msgs::msg::Pose & goal,
    nav_msgs::msg::Path & plan);

  rcl_interfaces::msg::SetParametersResult
  dynamicParametersCallback(const std::vector<rclcpp::Parameter> & parameters);

  inline double squared_distance(
    const geometry_msgs::msg::Pose & p1,
    const geometry_msgs::msg::Pose & p2)
  {
    double dx = p1.position.x - p2.position.x;
    double dy = p1.position.y - p2.position.y;
    return dx * dx + dy * dy;
  }

  // ROS interfaces
  nav2_util::LifecycleNode::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;
  rclcpp::Logger logger_{rclcpp::get_logger("DStarPlanner")};
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;

  // Parameters
  std::string global_frame_, name_;
  double tolerance_{0.5};
  bool use_final_approach_orientation_{true};

};

}  // namespace d_star_planner

#endif  // D_STAR_PLANNER__D_STAR_PLANNER_HPP_