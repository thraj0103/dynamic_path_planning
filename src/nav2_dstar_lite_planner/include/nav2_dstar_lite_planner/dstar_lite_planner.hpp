// dstar_lite_planner.hpp

#ifndef NAV2_DSTAR_LITE_PLANNER_HPP_
#define NAV2_DSTAR_LITE_PLANNER_HPP_

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"

namespace nav2_dstar_lite_planner
{

struct Cell {
  int x, y;
  double g;  // g-value (estimate of goal distance)
  double rhs; // rhs-value (one-step lookahead based on g-values)
  std::array<double, 2> key; // key values for priority queue ordering
  
  Cell() : x(0), y(0), g(INFINITY), rhs(INFINITY), key({INFINITY, INFINITY}) {}
  Cell(int x_coord, int y_coord) : x(x_coord), y(y_coord), g(INFINITY), rhs(INFINITY), key({INFINITY, INFINITY}) {}
};

// Custom comparator for the priority queue
struct CellComparator {
  bool operator()(const Cell& lhs, const Cell& rhs) const {
    // Lexicographic ordering of keys
    if (lhs.key[0] > rhs.key[0]) return true;
    if (lhs.key[0] < rhs.key[0]) return false;
    return lhs.key[1] > rhs.key[1];
  }
};

class DStarLitePlanner : public nav2_core::GlobalPlanner
{
public:
  DStarLitePlanner();
  ~DStarLitePlanner();

  // Required overrides from GlobalPlanner interface
  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
    std::string name, 
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  // D* Lite specific methods
  std::array<double, 2> calculateKey(const Cell& cell);
  void initialize();
  void updateVertex(Cell& cell);
  void computeShortestPath();
  
  // Helper methods
  double calculateHeuristic(const Cell& a, const Cell& b);
  bool isGoal(const Cell& cell) const;
  bool isStart(const Cell& cell) const;
  std::vector<Cell> getNeighbors(const Cell& cell);
  double getCost(const Cell& from, const Cell& to);
  Cell worldToGrid(const geometry_msgs::msg::PoseStamped& pose);
  geometry_msgs::msg::PoseStamped gridToWorld(const Cell& cell);
  nav_msgs::msg::Path extractPath();
  void clearData();
  void updateObstacleArea(const Cell& obstacle_cell);
  void storePath(const nav_msgs::msg::Path& path);
  
  // Class members
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D* costmap_;
  std::string global_frame_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("DStarLitePlanner")};
  
  // Cell storage and priority queue
  std::priority_queue<Cell, std::vector<Cell>, CellComparator> open_list_;
  std::unordered_map<int, std::unordered_map<int, Cell>> grid_map_;
  
  // Start and goal positions
  Cell start_cell_;
  Cell goal_cell_;
  double k_m_; // Accumulated heuristic change to avoid priority queue reordering
  bool first_planning_; // Track if this is the first planning cycle
  std::vector<Cell> previous_path_;
  
  // Parameters
  double inflation_radius_;
  bool allow_unknown_;
  int max_iterations_;
};

}  // namespace nav2_dstar_lite_planner

#endif  // NAV2_DSTAR_LITE_PLANNER_HPP_