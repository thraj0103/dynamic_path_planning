cmake_minimum_required(VERSION 3.5)
project(nav2_dstar_lite_planner)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(nav2_common REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)

nav2_package()

include_directories(
  include
)

set(library_name ${PROJECT_NAME})

add_library(${library_name} SHARED
  src/dstar_lite_planner.cpp
)

ament_target_dependencies(${library_name}
  rclcpp
  rclcpp_lifecycle
  nav2_util
  nav2_core
  nav_msgs
  nav2_costmap_2d
  pluginlib
  tf2_ros
)

pluginlib_export_plugin_description_file(nav2_core nav2_dstar_lite_planner_plugin.xml)

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(FILES nav2_dstar_lite_planner_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  nav2_util
  nav2_core
  nav_msgs
  nav2_costmap_2d
  pluginlib
)

ament_package()