/*
 * ov_landmap: descriptor-free persistent landmark map.
 */

#include "map/UpdaterMap.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

namespace ov_landmap {

namespace {
constexpr size_t ICP_MIN_CORRESPONDENCES = 3;
constexpr int ICP_MAX_ITERATIONS = 200;
constexpr double ICP_TRANSFORMATION_EPS = 1e-10;
constexpr double ICP_EUCLIDEAN_FITNESS_EPS = 1e-8;
} // namespace

UpdaterMap::UpdaterMap() : options_(Options()) {}

UpdaterMap::UpdaterMap(Options options) : options_(options) {}

Eigen::Isometry3d UpdaterMap::process_landmarks(std::vector<TrackedLandmark> landmarks, const Eigen::Isometry3d &current_pose_from_ekf) {
  (void)landmarks;
  return current_pose_from_ekf;
}

size_t UpdaterMap::find_map_landmark_index(size_t map_landmark_id) const {
  if (map_landmark_id == INVALID_MAP_LANDMARK_ID) {
    return INVALID_MAP_LANDMARK_ID;
  }
  for (size_t i = 0; i < landmarks_.size(); ++i) {
    if (landmarks_.at(i).map_id == map_landmark_id) {
      return i;
    }
  }
  return INVALID_MAP_LANDMARK_ID;
}

float UpdaterMap::estimate_rigid_transform(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &dst,
                                           Eigen::Isometry3d &T_dst_src, double &mean_error_m, double &max_error_m) {
  mean_error_m = -1.0;
  max_error_m = -1.0;

  if (src.size() < ICP_MIN_CORRESPONDENCES || dst.size() < ICP_MIN_CORRESPONDENCES || !T_dst_src.matrix().allFinite()) {
    return 0.0f;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dst(new pcl::PointCloud<pcl::PointXYZ>());
  cloud_src->reserve(src.size());
  cloud_dst->reserve(dst.size());

  for (const Eigen::Vector3d &point : src) {
    if (point.allFinite()) {
      cloud_src->push_back(pcl::PointXYZ(static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z())));
    }
  }

  for (const Eigen::Vector3d &point : dst) {
    if (point.allFinite()) {
      cloud_dst->push_back(pcl::PointXYZ(static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z())));
    }
  }

  if (cloud_src->size() < ICP_MIN_CORRESPONDENCES || cloud_dst->size() < ICP_MIN_CORRESPONDENCES) {
    return 0.0f;
  }

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud_src);
  icp.setInputTarget(cloud_dst);
  icp.setMaximumIterations(ICP_MAX_ITERATIONS);
  icp.setTransformationEpsilon(ICP_TRANSFORMATION_EPS);
  icp.setEuclideanFitnessEpsilon(ICP_EUCLIDEAN_FITNESS_EPS);

  Eigen::Matrix4f initial_guess = T_dst_src.matrix().cast<float>();
  pcl::PointCloud<pcl::PointXYZ> aligned;
  icp.align(aligned, initial_guess);

  if (!icp.hasConverged()) {
    return 0.0f;
  }

  const Eigen::Matrix4f final_transform = icp.getFinalTransformation();
  if (!final_transform.allFinite()) {
    return 0.0f;
  }

  T_dst_src.matrix() = final_transform.cast<double>();
  mean_error_m = 0.0;
  max_error_m = 0.0;
  for (const Eigen::Vector3d &point_src : src) {
    if (!point_src.allFinite()) {
      continue;
    }
    const Eigen::Vector3d point_transformed = T_dst_src * point_src;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const Eigen::Vector3d &point_dst : dst) {
      if (!point_dst.allFinite()) {
        continue;
      }
      best_distance = std::min(best_distance, (point_transformed - point_dst).norm());
    }
    if (!std::isfinite(best_distance)) {
      continue;
    }
    mean_error_m += best_distance;
    max_error_m = std::max(max_error_m, best_distance);
  }
  mean_error_m /= static_cast<double>(cloud_src->size());

  if (!std::isfinite(mean_error_m) || !std::isfinite(max_error_m) || !T_dst_src.matrix().allFinite()) {
    return 0.0f;
  }
  return 1.0f;
}

} // namespace ov_landmap
