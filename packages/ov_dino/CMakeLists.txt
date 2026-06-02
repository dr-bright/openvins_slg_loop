cmake_minimum_required(VERSION 3.3)
project(ov_dino)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
endif ()

option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin QUIET COMPONENTS roscpp rosbag sensor_msgs cv_bridge ov_core)
find_package(ament_cmake QUIET)

if (catkin_FOUND AND ENABLE_ROS)
  add_definitions(-DROS_AVAILABLE=1)
  catkin_package(
    INCLUDE_DIRS src
    CATKIN_DEPENDS roscpp rosbag sensor_msgs cv_bridge ov_core
    LIBRARIES ${PROJECT_NAME}
  )
  include_directories(
    src
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )

  add_library(${PROJECT_NAME}
    src/track/dino_backend.cpp
    src/track/FeatureDatabaseDINO.cpp
    src/track/TrackDINOConfig.cpp
    src/track/TrackDINO.cpp
  )
  target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )

  add_executable(dino_backend_smoke_test
    tests/dino_backend_smoke_test.cpp
  )
  target_link_libraries(dino_backend_smoke_test
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )

  add_executable(dino_client_image_gui
    tests/dino_client_image_gui.cpp
  )
  target_link_libraries(dino_client_image_gui
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )

  add_executable(dino_backend_image_gui
    tests/dino_backend_image_gui.cpp
  )
  target_link_libraries(dino_backend_image_gui
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )

  add_executable(evaluate_tracker_dino
    tests/evaluate_tracker_dino.cpp
  )
  target_link_libraries(evaluate_tracker_dino
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )
elseif (ament_cmake_FOUND AND ENABLE_ROS)
  find_package(rclcpp REQUIRED)
  find_package(ov_core REQUIRED)

  add_library(${PROJECT_NAME}
    src/track/dino_backend.cpp
    src/track/FeatureDatabaseDINO.cpp
    src/track/TrackDINOConfig.cpp
    src/track/TrackDINO.cpp
  )
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )
  ament_target_dependencies(${PROJECT_NAME} rclcpp ov_core)
  target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
  )

  install(TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(DIRECTORY src/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
  ament_export_include_directories(include)
  ament_export_libraries(${PROJECT_NAME})
  ament_export_dependencies(rclcpp ov_core)
  ament_package()
else ()
  message(STATUS "No ROS environment detected for ov_dino scaffold.")
endif ()
