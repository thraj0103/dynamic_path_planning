// dstar_lite_planner.cpp

#include "nav2_dstar_lite_planner/dstar_lite_planner.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <tf2/LinearMath/Quaternion.h>


PLUGINLIB_EXPORT_CLASS(nav2_dstar_lite_planner::DStarLitePlanner, nav2_core::GlobalPlanner)

namespace nav2_dstar_lite_planner
{

DStarLitePlanner::DStarLitePlanner()
: costmap_(nullptr),
  k_m_(0.0),
  first_planning_(true)
{
}

DStarLitePlanner::~DStarLitePlanner()
{
  RCLCPP_INFO(logger_, "Destroying plugin");
}

void DStarLitePlanner::configure(
  rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
  std::string name, 
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  
  // Using tf buffer is not needed in this implementation, but keep a reference
  // to avoid unused parameter warnings
  (void)tf;
  
  clock_ = parent->get_clock();
  logger_ = parent->get_logger();
  
  RCLCPP_INFO(logger_, "Configuring D* Lite planner");
  
  global_frame_ = costmap_ros_->getGlobalFrameID();

  // Get parameters
  nav2_util::declare_parameter_if_not_declared(
    parent, name + ".inflation_radius", rclcpp::ParameterValue(0.5));
  parent->get_parameter(name + ".inflation_radius", inflation_radius_);

  nav2_util::declare_parameter_if_not_declared(
    parent, name + ".allow_unknown", rclcpp::ParameterValue(true));
  parent->get_parameter(name + ".allow_unknown", allow_unknown_);

  nav2_util::declare_parameter_if_not_declared(
    parent, name + ".max_iterations", rclcpp::ParameterValue(1000000));
  parent->get_parameter(name + ".max_iterations", max_iterations_);
}

void DStarLitePlanner::activate()
{
  RCLCPP_INFO(logger_, "Activating");
}

void DStarLitePlanner::deactivate()
{
  RCLCPP_INFO(logger_, "Deactivating");
}

void DStarLitePlanner::cleanup()
{
  RCLCPP_INFO(logger_, "Cleaning up");
  clearData();
}

// Replace your existing createPlan method (around line 47)
nav_msgs::msg::Path DStarLitePlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  RCLCPP_INFO(logger_, "Creating plan from (%g, %g) to (%g, %g)",
    start.pose.position.x, start.pose.position.y,
    goal.pose.position.x, goal.pose.position.y);
  
  // Track whether this is the first planning request
  bool is_first_planning = first_planning_;
  
  // Get new start and goal cells
  Cell new_start_cell = worldToGrid(start);
  Cell new_goal_cell = worldToGrid(goal);
  
  // Handle first planning request
  if (is_first_planning) {
    // Full initialization for first plan
    clearData();
    start_cell_ = new_start_cell;
    goal_cell_ = new_goal_cell;
    initialize();
    first_planning_ = false;
  } else {
    // Check for changes in the environment and goal
    bool goal_changed = (new_goal_cell.x != goal_cell_.x || new_goal_cell.y != goal_cell_.y);
    
    if (goal_changed) {
      // If goal changed, reinitialize completely
      clearData();
      start_cell_ = new_start_cell;
      goal_cell_ = new_goal_cell;
      initialize();
    } else {
      // Keep existing data but update start position and check for obstacles
      start_cell_ = new_start_cell;
      
      // Scan for changes in the costmap along the previous path
      bool obstacles_changed = false;
      for (const auto& cell : previous_path_) {
        unsigned char current_cost = costmap_->getCost(cell.x, cell.y);
        if (current_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
          obstacles_changed = true;
          // Update the rhs-values of cells around the obstacle
          updateObstacleArea(cell);
        }
      }
      
      if (obstacles_changed) {
        RCLCPP_INFO(logger_, "Detected obstacle changes, replanning...");
      }
      
      // Update k_m for consistency when start changes
      k_m_ = k_m_ + calculateHeuristic(Cell(start_cell_.x, start_cell_.y), 
                                       Cell(new_start_cell.x, new_start_cell.y));
    }
  }
  
  // Compute shortest path with updated information
  computeShortestPath();
  
  // Extract path
  nav_msgs::msg::Path path = extractPath();
  
  // Store the path cells for obstacle detection in the next cycle
  storePath(path);
  
  // If path is empty, create a direct path (for testing)
  if (path.poses.empty() || path.poses.size() < 2) {
    RCLCPP_WARN(logger_, "D* Lite failed to find a path, creating a direct path for testing");
    path.poses.clear();
    
    // Create a straight-line path with waypoints
    int num_waypoints = 10;
    for (int i = 0; i <= num_waypoints; i++) {
      double ratio = static_cast<double>(i) / num_waypoints;
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = global_frame_;
      pose.header.stamp = clock_->now();
      pose.pose.position.x = start.pose.position.x + ratio * (goal.pose.position.x - start.pose.position.x);
      pose.pose.position.y = start.pose.position.y + ratio * (goal.pose.position.y - start.pose.position.y);
      pose.pose.position.z = 0.0;
      pose.pose.orientation = goal.pose.orientation;
      path.poses.push_back(pose);
    }
  }
  
  path.header.stamp = clock_->now();
  path.header.frame_id = global_frame_;
  
  RCLCPP_INFO(logger_, "Final path has %zu poses", path.poses.size());
  return path;
}
void DStarLitePlanner::clearData()
{
  // Clear the priority queue and grid cells
  while (!open_list_.empty()) {
    open_list_.pop();
  }
  grid_map_.clear();
  k_m_ = 0.0;
}

void DStarLitePlanner::initialize()
{
  // Clear previous data
  clearData();
  
  // Initialize goal state
  grid_map_[goal_cell_.x][goal_cell_.y] = goal_cell_;
  grid_map_[goal_cell_.x][goal_cell_.y].g = INFINITY;  // Initialize g to infinity
  grid_map_[goal_cell_.x][goal_cell_.y].rhs = 0.0;     // Goal has rhs = 0
  
  // Calculate the key for the goal
  grid_map_[goal_cell_.x][goal_cell_.y].key = calculateKey(grid_map_[goal_cell_.x][goal_cell_.y]);
  
  // Add goal to open list
  open_list_.push(grid_map_[goal_cell_.x][goal_cell_.y]);
  
  // Initialize start state
  grid_map_[start_cell_.x][start_cell_.y] = start_cell_;
  grid_map_[start_cell_.x][start_cell_.y].g = INFINITY;
  grid_map_[start_cell_.x][start_cell_.y].rhs = INFINITY;
  
  RCLCPP_INFO(logger_, "Initialized D* Lite with goal at (%d, %d) and start at (%d, %d)", 
              goal_cell_.x, goal_cell_.y, start_cell_.x, start_cell_.y);
  RCLCPP_INFO(logger_, "Goal cell: g=%f, rhs=%f, key=[%f, %f]", 
              grid_map_[goal_cell_.x][goal_cell_.y].g,
              grid_map_[goal_cell_.x][goal_cell_.y].rhs,
              grid_map_[goal_cell_.x][goal_cell_.y].key[0],
              grid_map_[goal_cell_.x][goal_cell_.y].key[1]);
}

std::array<double, 2> DStarLitePlanner::calculateKey(const Cell& cell)
{
  std::array<double, 2> key;
  
  // Calculate the minimum of g and rhs
  double min_g_rhs = std::min(cell.g, cell.rhs);
  
  // First component: min(g, rhs) + h(start, cell) + k_m
  key[0] = min_g_rhs + calculateHeuristic(cell, start_cell_) + k_m_;
  
  // Second component: min(g, rhs)
  key[1] = min_g_rhs;
  
  return key;
}

double DStarLitePlanner::calculateHeuristic(const Cell& a, const Cell& b)
{
  // Manhattan distance (admissible and consistent)
  return std::abs(a.x - b.x) + std::abs(a.y - b.y);
  
  // Or Euclidean distance
  // double dx = a.x - b.x;
  // double dy = a.y - b.y;
  // return std::sqrt(dx * dx + dy * dy);
}

void DStarLitePlanner::updateVertex(Cell& cell)
{
  // Don't update the goal vertex
  if (cell.x == goal_cell_.x && cell.y == goal_cell_.y) {
    return;
  }
  
  // Save old rhs value for comparison
  double old_rhs = cell.rhs;
  
  // Update rhs value (one-step lookahead)
  cell.rhs = INFINITY;
  
  // Get successors (cells this one can reach)
  std::vector<Cell> neighbors = getNeighbors(cell);
  
  // Compute new rhs value - based on neighbor g-values
  for (const auto& neighbor : neighbors) {
    int nx = neighbor.x;
    int ny = neighbor.y;
    
    if (grid_map_.find(nx) != grid_map_.end() && 
        grid_map_[nx].find(ny) != grid_map_[nx].end()) {
      
      double cost = getCost(cell, grid_map_[nx][ny]);
      
      if (cost < INFINITY) {
        double tentative_rhs = grid_map_[nx][ny].g + cost;
        
        if (tentative_rhs < cell.rhs) {
          cell.rhs = tentative_rhs;
          RCLCPP_DEBUG(logger_, "Cell (%d, %d) rhs updated to %f via neighbor (%d, %d) with g=%f",
                      cell.x, cell.y, cell.rhs, nx, ny, grid_map_[nx][ny].g);
        }
      }
    }
  }
  
  // Log significant rhs changes
  if (fabs(old_rhs - cell.rhs) > 1.0) {
    RCLCPP_DEBUG(logger_, "Cell (%d, %d) rhs changed significantly from %f to %f",
                cell.x, cell.y, old_rhs, cell.rhs);
  }
  
  // If the cell is inconsistent, add it to the open list
  if (cell.g != cell.rhs) {
    // Update key
    cell.key = calculateKey(cell);
    open_list_.push(cell);
  }
}

void DStarLitePlanner::computeShortestPath()
{
  int iterations = 0;
  
  RCLCPP_INFO(logger_, "Starting D* Lite computation with %zu cells in queue", open_list_.size());
  
  // Process until the start cell's key is >= the top key in the open list
  // AND the start cell is consistent (g = rhs)
  while (!open_list_.empty() && 
         ((open_list_.top().key[0] < calculateKey(grid_map_[start_cell_.x][start_cell_.y])[0]) ||
          (open_list_.top().key[0] == calculateKey(grid_map_[start_cell_.x][start_cell_.y])[0] && 
           open_list_.top().key[1] < calculateKey(grid_map_[start_cell_.x][start_cell_.y])[1]) ||
          (grid_map_[start_cell_.x][start_cell_.y].rhs != grid_map_[start_cell_.x][start_cell_.y].g)) && 
         iterations < max_iterations_) {
    
    // Get the cell with the minimum key
    Cell current = open_list_.top();
    open_list_.pop();
    
    // Track expansion for debugging
    if (iterations % 1000 == 0) {
      RCLCPP_DEBUG(logger_, "Expanding cell (%d, %d) with key [%f, %f], g=%f, rhs=%f (iteration %d)",
                  current.x, current.y, current.key[0], current.key[1], 
                  current.g, current.rhs, iterations);
    }
    
    // Get the cell from the grid map
    Cell& cell = grid_map_[current.x][current.y];
    
    // Calculate new key using current k_m_
    std::array<double, 2> new_key = calculateKey(cell);
    
    // If the stored key is outdated, update and reinsert
    if (current.key[0] > new_key[0] || 
        (current.key[0] == new_key[0] && current.key[1] > new_key[1])) {
      cell.key = new_key;
      open_list_.push(cell);
      iterations++;
      continue;
    }
    
    // If cell is overconsistent, make it consistent
    if (cell.g > cell.rhs) {
      cell.g = cell.rhs;
      
      // Update predecessors (cells that point to this one)
      std::vector<Cell> neighbors = getNeighbors(cell);
      for (auto& neighbor : neighbors) {
        int nx = neighbor.x;
        int ny = neighbor.y;
        
        // Ensure the neighbor exists in the grid map
        if (grid_map_.find(nx) == grid_map_.end() || 
            grid_map_[nx].find(ny) == grid_map_[nx].end()) {
          grid_map_[nx][ny] = neighbor;
        }
        
        // Update the neighbor's rhs-value if needed
        updateVertex(grid_map_[nx][ny]);
      }
    }
    // If cell is underconsistent, make it overconsistent
    else if (cell.g < cell.rhs) {
      cell.g = INFINITY;
      
      // Update the current cell and its predecessors
      std::vector<Cell> neighbors = getNeighbors(cell);
      neighbors.push_back(cell);  // Include the cell itself
      
      for (auto& neighbor : neighbors) {
        int nx = neighbor.x;
        int ny = neighbor.y;
        
        // Ensure the neighbor exists in the grid map
        if (grid_map_.find(nx) == grid_map_.end() || 
            grid_map_[nx].find(ny) == grid_map_[nx].end()) {
          grid_map_[nx][ny] = neighbor;
        }
        
        // Update the neighbor's rhs-value
        updateVertex(grid_map_[nx][ny]);
      }
    }
    
    iterations++;
    if (grid_map_[start_cell_.x][start_cell_.y].rhs < INFINITY) {
      RCLCPP_INFO(logger_, "Found path with cost %f", grid_map_[start_cell_.x][start_cell_.y].rhs);
    } else {
      RCLCPP_WARN(logger_, "No path found - start cell has infinite rhs value");
    }

    // Print progress periodically
    if (iterations % 10000 == 0) {
      RCLCPP_INFO(logger_, "D* Lite: Processed %d iterations", iterations);
      RCLCPP_INFO(logger_, "Start cell: g=%f, rhs=%f", 
                 grid_map_[start_cell_.x][start_cell_.y].g,
                 grid_map_[start_cell_.x][start_cell_.y].rhs);
      
      // Check if we're close to a solution
      if (grid_map_[start_cell_.x][start_cell_.y].rhs < INFINITY) {
        RCLCPP_INFO(logger_, "Start cell has a finite rhs value: %f", 
                   grid_map_[start_cell_.x][start_cell_.y].rhs);
      }
    }
  }
  
  if (iterations >= max_iterations_) {
    RCLCPP_WARN(logger_, "D* Lite reached maximum iterations (%d)", max_iterations_);
  }
  
  RCLCPP_INFO(logger_, "D* Lite finished after %d iterations", iterations);
  RCLCPP_INFO(logger_, "Start cell (final): g=%f, rhs=%f", 
             grid_map_[start_cell_.x][start_cell_.y].g,
             grid_map_[start_cell_.x][start_cell_.y].rhs);
}

std::vector<Cell> DStarLitePlanner::getNeighbors(const Cell& cell)
{
  std::vector<Cell> neighbors;
  
  // 8-connected grid
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) continue; // Skip the cell itself
      
      int nx = cell.x + dx;
      int ny = cell.y + dy;
      
      // Check bounds
      if (nx < 0 || nx >= static_cast<int>(costmap_->getSizeInCellsX()) ||
          ny < 0 || ny >= static_cast<int>(costmap_->getSizeInCellsY())) {
        continue;
      }
      
      // Create the neighbor cell
      Cell neighbor(nx, ny);
      
      // Check if already in grid_map
      if (grid_map_.find(nx) != grid_map_.end() && 
          grid_map_[nx].find(ny) != grid_map_[nx].end()) {
        neighbor = grid_map_[nx][ny];
      }
      
      neighbors.push_back(neighbor);
    }
  }
  
  return neighbors;
}

double DStarLitePlanner::getCost(const Cell& from, const Cell& to)
{
  // Check boundary constraints
  if (to.x < 0 || to.x >= static_cast<int>(costmap_->getSizeInCellsX()) ||
      to.y < 0 || to.y >= static_cast<int>(costmap_->getSizeInCellsY())) {
    return INFINITY;
  }
  
  // Get cost from costmap
  unsigned char cost = costmap_->getCost(to.x, to.y);
  
  // Only treat truly lethal obstacles as impassable
  if (cost == nav2_costmap_2d::LETHAL_OBSTACLE) {
    return INFINITY;
  }
  
  // Calculate the base movement cost (diagonal movements cost more)
  double dx = to.x - from.x;
  double dy = to.y - from.y;
  double movement_cost = std::sqrt(dx * dx + dy * dy);
  
  // Handle different cost regions
  if (cost == nav2_costmap_2d::FREE_SPACE) {
    // Free space - normal cost
    return movement_cost;
  } else if (cost == nav2_costmap_2d::NO_INFORMATION && !allow_unknown_) {
    // Unknown space not allowed
    return INFINITY;
  } else if (cost == nav2_costmap_2d::NO_INFORMATION && allow_unknown_) {
    // Unknown space allowed but with higher cost
    return movement_cost * 2.0;
  } else if (cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
    // Inside the robot's footprint - high cost but not impossible
    // This allows planning through tight spaces when necessary
    return movement_cost * 4.0;
  } else {
    // Inflation zone - cost increases with proximity to obstacles
    // Map the range 1-252 to a cost multiplier of 1.0-5.0
    double cost_factor = 1.0 + 4.0 * (static_cast<double>(cost) / 252.0);
    return movement_cost * cost_factor;
  }
}
bool DStarLitePlanner::isGoal(const Cell& cell) const
{
  return cell.x == goal_cell_.x && cell.y == goal_cell_.y;
}

bool DStarLitePlanner::isStart(const Cell& cell) const
{
  return cell.x == start_cell_.x && cell.y == start_cell_.y;
}

Cell DStarLitePlanner::worldToGrid(const geometry_msgs::msg::PoseStamped& pose)
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(pose.pose.position.x, pose.pose.position.y, mx, my)) {
    RCLCPP_ERROR(logger_, "Cannot convert world coordinates (%f, %f) to map coordinates!",
      pose.pose.position.x, pose.pose.position.y);
    // Return a cell at the origin as fallback
    return Cell(0, 0);
  }
  
  RCLCPP_INFO(logger_, "Converting world (%f, %f) to grid (%d, %d)",
    pose.pose.position.x, pose.pose.position.y, static_cast<int>(mx), static_cast<int>(my));
  
  Cell cell(static_cast<int>(mx), static_cast<int>(my));
  return cell;
}

geometry_msgs::msg::PoseStamped DStarLitePlanner::gridToWorld(const Cell& cell)
{
  double wx, wy;
  costmap_->mapToWorld(cell.x, cell.y, wx, wy);
  
  geometry_msgs::msg::PoseStamped pose;
  pose.pose.position.x = wx;
  pose.pose.position.y = wy;
  pose.pose.position.z = 0.0;
  pose.pose.orientation.w = 1.0;
  
  return pose;
}

nav_msgs::msg::Path DStarLitePlanner::extractPath()
{
  nav_msgs::msg::Path path;
  std::vector<geometry_msgs::msg::PoseStamped> poses;
  
  // Check if we have a valid path
  if (grid_map_[start_cell_.x][start_cell_.y].rhs == INFINITY) {
    RCLCPP_WARN(logger_, "No valid path found! Start cell has infinite rhs-value.");
    return path;
  }
  
  // Start from the start cell
  Cell current = start_cell_;
  
  // Add start pose
  poses.push_back(gridToWorld(current));
  RCLCPP_DEBUG(logger_, "Adding start cell (%d, %d) to path", current.x, current.y);
  
  // Follow the path to the goal
  int max_steps = costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY();
  max_steps = std::min(max_steps, 1000);  // Cap at 1000 to prevent excessive paths
  
  int step_count = 0;
  
  while (!isGoal(current) && step_count < max_steps) {
    // Find the best next cell
    std::vector<Cell> neighbors = getNeighbors(current);
    
    Cell next_cell = current;  // Default to current if no better neighbor found
    double min_cost = INFINITY;
    
    for (const auto& neighbor : neighbors) {
      int nx = neighbor.x;
      int ny = neighbor.y;
      
      // Skip if not in grid_map
      if (grid_map_.find(nx) == grid_map_.end() || 
          grid_map_[nx].find(ny) == grid_map_[nx].end()) {
        continue;
      }
      
      // Skip if cost to neighbor is infinite
      double cost_to_neighbor = getCost(current, grid_map_[nx][ny]);
      if (cost_to_neighbor == INFINITY) {
        continue;
      }
      
      // Calculate cost through this neighbor
      double total_cost = grid_map_[nx][ny].g + cost_to_neighbor;
      
      if (total_cost < min_cost) {
        min_cost = total_cost;
        next_cell = grid_map_[nx][ny];
      }
    }
    
    // Check if we're stuck in a local minimum
    if (next_cell.x == current.x && next_cell.y == current.y) {
      RCLCPP_WARN(logger_, "Stuck in local minimum at (%d, %d), terminating path extraction",
                 current.x, current.y);
      break;
    }
    
    // Move to next cell
    current = next_cell;
    poses.push_back(gridToWorld(current));
    
    step_count++;
    
    // Add points with adaptive density - fewer points when far from obstacles
    if (step_count % 5 == 0) {
      // Check nearby for obstacles
      bool near_obstacle = false;
      for (const auto& neighbor : getNeighbors(current)) {
        unsigned char cost = costmap_->getCost(neighbor.x, neighbor.y);
        if (cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
          near_obstacle = true;
          break;
        }
      }
      
      // Skip a few steps if we're in open space
      if (!near_obstacle && !isGoal(next_cell) && 
          calculateHeuristic(current, goal_cell_) > 5.0) {
        // Jump ahead a bit
        for (int i = 0; i < 2; i++) {
          std::vector<Cell> far_neighbors = getNeighbors(current);
          Cell best_neighbor = current;
          double min_cost = INFINITY;
          
          for (const auto& neighbor : far_neighbors) {
            if (grid_map_.find(neighbor.x) == grid_map_.end() || 
                grid_map_[neighbor.x].find(neighbor.y) == grid_map_[neighbor.x].end()) {
              continue;
            }
            
            double cost = grid_map_[neighbor.x][neighbor.y].g;
            if (cost < min_cost) {
              min_cost = cost;
              best_neighbor = grid_map_[neighbor.x][neighbor.y];
            }
          }
          
          if (best_neighbor.x != current.x || best_neighbor.y != current.y) {
            current = best_neighbor;
            step_count++;
          } else {
            break;
          }
        }
      }
    }
  }
  
  // Check if we reached the goal
  if (isGoal(current)) {
    RCLCPP_INFO(logger_, "Path successfully extracted to goal");
  } else {
    RCLCPP_WARN(logger_, "Failed to reach goal in path extraction after %d steps", step_count);
    
    // Add goal pose if not already at the goal and we didn't reach the step limit
    if (step_count < max_steps) {
      poses.push_back(gridToWorld(goal_cell_));
    }
  }
  
  // Add path smoothing to reduce jaggedness
  if (poses.size() > 3) {
    std::vector<geometry_msgs::msg::PoseStamped> smoothed_path;
    smoothed_path.push_back(poses.front());
    
    for (size_t i = 1; i < poses.size() - 1; i++) {
      // Skip some points for smoothing, but keep those near obstacles
      unsigned char cost = costmap_->getCost(worldToGrid(poses[i]).x, worldToGrid(poses[i]).y);
      if (i % 2 == 0 && cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        continue;
      }
      
      // Calculate orientation based on direction to next point
      if (i > 0 && i < poses.size() - 1) {
        double dx = poses[i+1].pose.position.x - poses[i-1].pose.position.x;
        double dy = poses[i+1].pose.position.y - poses[i-1].pose.position.y;
        double yaw = std::atan2(dy, dx);
        
        // Manually set the quaternion values for orientation
        double cy = std::cos(yaw * 0.5);
        double sy = std::sin(yaw * 0.5);
        poses[i].pose.orientation.x = 0.0;
        poses[i].pose.orientation.y = 0.0;
        poses[i].pose.orientation.z = sy;
        poses[i].pose.orientation.w = cy;
      }
      
      smoothed_path.push_back(poses[i]);
    }
    
    smoothed_path.push_back(poses.back());
    poses = smoothed_path;
  }
  
  path.poses = poses;
  RCLCPP_INFO(logger_, "Extracted path with %zu poses", poses.size());
  
  return path;
}

// Add these two methods at the end of the file, right before the closing namespace (around line 550)
void DStarLitePlanner::updateObstacleArea(const Cell& obstacle_cell)
{
  // Update a 3x3 area around the obstacle
  RCLCPP_INFO(logger_, "Updating area around obstacle at (%d, %d)", 
             obstacle_cell.x, obstacle_cell.y);
             
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      int nx = obstacle_cell.x + dx;
      int ny = obstacle_cell.y + dy;
      
      // Check bounds
      if (nx < 0 || nx >= static_cast<int>(costmap_->getSizeInCellsX()) ||
          ny < 0 || ny >= static_cast<int>(costmap_->getSizeInCellsY())) {
        continue;
      }
      
      // Ensure the cell exists in the grid map
      if (grid_map_.find(nx) == grid_map_.end() || 
          grid_map_[nx].find(ny) == grid_map_[nx].end()) {
        grid_map_[nx][ny] = Cell(nx, ny);
      }
      
      // Update the cell's rhs-value
      updateVertex(grid_map_[nx][ny]);
    }
  }
}

void DStarLitePlanner::storePath(const nav_msgs::msg::Path& path)
{
  previous_path_.clear();
  
  // Store a downsampled version of the path to reduce computational load
  // Aim for approximately 100-200 points maximum
  int stride = std::max(1, static_cast<int>(path.poses.size() / 100));
  
  RCLCPP_INFO(logger_, "Storing path with %zu points (sampling every %d points)", 
              path.poses.size(), stride);
  
  for (size_t i = 0; i < path.poses.size(); i += stride) {
    Cell cell = worldToGrid(path.poses[i]);
    previous_path_.push_back(cell);
  }
  
  // Ensure the goal is included
  if (!path.poses.empty()) {
    Cell goal_cell = worldToGrid(path.poses.back());
    if (previous_path_.empty() || 
        previous_path_.back().x != goal_cell.x || 
        previous_path_.back().y != goal_cell.y) {
      previous_path_.push_back(goal_cell);
    }
  }
  
  RCLCPP_INFO(logger_, "Stored downsampled path with %zu points for obstacle detection", 
              previous_path_.size());
}


}  // namespace nav2_dstar_lite_planner