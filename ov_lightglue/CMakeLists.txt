cmake_minimum_required(VERSION 3.3)
project(ov_lightglue)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
endif ()
find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)

option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin QUIET COMPONENTS roscpp rosbag geometry_msgs nav_msgs sensor_msgs cv_bridge ov_core)
find_package(ament_cmake QUIET)

if (catkin_FOUND AND ENABLE_ROS)
  add_definitions(-DROS_AVAILABLE=1)
  catkin_package(
    INCLUDE_DIRS src
    CATKIN_DEPENDS roscpp rosbag geometry_msgs nav_msgs sensor_msgs cv_bridge ov_core
    LIBRARIES ${PROJECT_NAME}
  )
  include_directories(
    src
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )

  add_library(${PROJECT_NAME}
    src/track/slg_backend.cpp
    src/track/FeatureSLG.cpp
    src/track/FeatureDatabaseSLG.cpp
    src/track/TrackSLGConfig.cpp
    src/track/TrackSLG.cpp
  )
  target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(slg_backend_smoke_test
    tests/slg_backend_smoke_test.cpp
  )
  target_link_libraries(slg_backend_smoke_test
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(slg_backend_smoke_test_vis
    tests/slg_backend_smoke_test_vis.cpp
  )
  target_link_libraries(slg_backend_smoke_test_vis
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(slg_backend_smoke_test_hist
    tests/slg_backend_smoke_test_hist.cpp
  )
  target_link_libraries(slg_backend_smoke_test_hist
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(onnxrt_check
    tests/onnxrt_check.cpp
  )
  target_link_libraries(onnxrt_check
    ${ONNXRUNTIME_LIB}
  )

  add_executable(track_slg_smoke_test
    tests/track_slg_smoke_test.cpp
  )
  target_link_libraries(track_slg_smoke_test
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(evaluate_tracker_slg
    src/evaluate_tracker.cpp
  )
  target_compile_definitions(evaluate_tracker_slg PRIVATE OVLG_TRACKER_SLG=1)
  target_link_libraries(evaluate_tracker_slg
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(evaluate_tracker_klt
    src/evaluate_tracker.cpp
  )
  target_compile_definitions(evaluate_tracker_klt PRIVATE OVLG_TRACKER_KLT=1)
  target_link_libraries(evaluate_tracker_klt
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(optimize_confidence
    src/optimize_confidence.cpp
  )
  target_link_libraries(optimize_confidence
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(evaluate_estimator
    src/evaluate_estimator.cpp
  )
  target_link_libraries(evaluate_estimator
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )

  add_executable(pose_to_file
    src/pose_to_file.cpp
  )
  target_link_libraries(pose_to_file
    ${catkin_LIBRARIES}
  )
elseif (ament_cmake_FOUND AND ENABLE_ROS)
  find_package(rclcpp REQUIRED)
  find_package(ov_core REQUIRED)

  add_library(${PROJECT_NAME}
    src/track/slg_backend.cpp
    src/track/FeatureSLG.cpp
    src/track/FeatureDatabaseSLG.cpp
    src/track/TrackSLGConfig.cpp
    src/track/TrackSLG.cpp
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
    ${ONNXRUNTIME_LIB}
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
  message(STATUS "No ROS environment detected for ov_lightglue scaffold.")
endif ()
