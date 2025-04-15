#include "d_star_planner/d_star_planner.hpp"

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <memory>

namespace d_star_planner
{

DStarPlanner::DStarPlanner() {}

DStarPlanner::~DStarPlanner()
{
  RCLCPP_INFO(logger_, "Destroying plugin %s of type DStarPlanner", name_.c_str());
}

void DStarPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros->getCostmap();
  global_frame_ = costmap_ros->getGlobalFrameID();
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  RCLCPP_INFO(logger_, "Configuring plugin %s of type DStarPlanner", name_.c_str());

  nav2_util::declare_parameter_if_not_declared(node, name + ".tolerance", rclcpp::ParameterValue(0.5));
  node->get_parameter(name + ".tolerance", tolerance_);

  nav2_util::declare_parameter_if_not_declared(node, name + ".use_final_approach_orientation", rclcpp::ParameterValue(true));
  node->get_parameter(name + ".use_final_approach_orientation", use_final_approach_orientation_);
}

void DStarPlanner::activate()
{
  auto node = node_.lock();
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(&DStarPlanner::dynamicParametersCallback, this, std::placeholders::_1));
}

void DStarPlanner::deactivate()
{
  auto node = node_.lock();
  if (dyn_params_handler_ && node) {
    node->remove_on_set_parameters_callback(dyn_params_handler_.get());
  }
  dyn_params_handler_.reset();
}

void DStarPlanner::cleanup() {}

rcl_interfaces::msg::SetParametersResult
DStarPlanner::dynamicParametersCallback(const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  for (auto param : parameters) {
    if (param.get_name() == name_ + ".tolerance") {
      tolerance_ = param.as_double();
    } else if (param.get_name() == name_ + ".use_final_approach_orientation") {
      use_final_approach_orientation_ = param.as_bool();
    }
  }

  return result;
}

nav_msgs::msg::Path DStarPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path path;
  path.header.stamp = clock_->now();
  path.header.frame_id = global_frame_;

  if (start.header.frame_id != global_frame_ || goal.header.frame_id != global_frame_) {
    RCLCPP_ERROR(logger_, "Start or goal pose is not in the global frame: %s", global_frame_.c_str());
    return path;
  }

  unsigned int mx_start, my_start, mx_goal, my_goal;
  if (!worldToMap(start.pose.position.x, start.pose.position.y, mx_start, my_start)) {
    RCLCPP_ERROR(logger_, "Start position is outside the costmap.");
    return path;
  }

  if (!worldToMap(goal.pose.position.x, goal.pose.position.y, mx_goal, my_goal)) {
    RCLCPP_ERROR(logger_, "Goal position is outside the costmap.");
    return path;
  }

  clearRobotCell(mx_start, my_start);

  auto heuristic = [](Cell a, Cell b) {
    return std::hypot(a.x - b.x, a.y - b.y);
  };

  Cell start_cell{static_cast<int>(mx_start), static_cast<int>(my_start)};
  Cell goal_cell{static_cast<int>(mx_goal), static_cast<int>(my_goal)};

  int width = costmap_->getSizeInCellsX();
  int height = costmap_->getSizeInCellsY();
  auto charmap = costmap_->getCharMap();
  std::vector<unsigned char> costmap_data(charmap, charmap + width * height);

  DStarLite planner(width, height, heuristic);
  planner.setCostmap(costmap_data, width, height);

  planner.initialize(start_cell, goal_cell);
  planner.computeShortestPath();

  auto cell_path = planner.extractPath();
  double resolution = costmap_->getResolution();
  double origin_x = costmap_->getOriginX();
  double origin_y = costmap_->getOriginY();

  for (size_t i = 0; i < cell_path.size(); ++i) {
    const auto & cell = cell_path[i];

    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = origin_x + (cell.x + 0.5) * resolution;
    pose.pose.position.y = origin_y + (cell.y + 0.5) * resolution;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;

    path.poses.push_back(pose);
  }

  // Final orientation interpolation
  if (use_final_approach_orientation_ && path.poses.size() >= 2) {
    auto & last = path.poses.back();
    auto & before_last = path.poses[path.poses.size() - 2];

    double dx = last.pose.position.x - before_last.pose.position.x;
    double dy = last.pose.position.y - before_last.pose.position.y;
    double theta = std::atan2(dy, dx);

    last.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(theta);
  }

  return path;
}

bool DStarPlanner::worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my)
{
  if (wx < costmap_->getOriginX() || wy < costmap_->getOriginY()) return false;

  mx = static_cast<unsigned int>((wx - costmap_->getOriginX()) / costmap_->getResolution());
  my = static_cast<unsigned int>((wy - costmap_->getOriginY()) / costmap_->getResolution());

  return mx < costmap_->getSizeInCellsX() && my < costmap_->getSizeInCellsY();
}

void DStarPlanner::clearRobotCell(unsigned int mx, unsigned int my)
{
  costmap_->setCost(mx, my, nav2_costmap_2d::FREE_SPACE);
}

}  // namespace d_star_planner

PLUGINLIB_EXPORT_CLASS(d_star_planner::DStarPlanner, nav2_core::GlobalPlanner)