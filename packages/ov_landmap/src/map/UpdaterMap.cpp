/*
 * ov_landmap: descriptor-free persistent landmark map.
 */

#include "map/UpdaterMap.h"

#include "types/Landmark.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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

UpdaterMap::UpdaterMap(std::shared_ptr<ov_core::TrackBase> tracker, Options options) : options_(options), tracker_(std::move(tracker)) {}

Eigen::Isometry3d UpdaterMap::process_landmarks(std::vector<TrackedLandmark> landmarks, const Eigen::Isometry3d &current_pose_from_ekf) {
  Eigen::Isometry3d T_M_G_initial = map_to_global_.transform.inverse();
  Eigen::Isometry3d T_M_G_estimate = T_M_G_initial;
  Eigen::Isometry3d localized_pose_from_map = current_pose_from_ekf;
  bool icp_succeeded = false;
  bool localization_agrees_with_odom = true;
  const bool can_estimate_transform = map_to_global_.measurements < options_.transform_estimation_cap;

  if (can_estimate_transform && landmarks_.size() >= options_.min_map_landmarks_for_icp) {
    std::vector<Eigen::Vector3d> points_global;
    std::vector<Eigen::Vector3d> points_map;
    points_global.reserve(landmarks.size());
    points_map.reserve(landmarks_.size());

    for (const TrackedLandmark &landmark : landmarks) {
      if (landmark.p_FinG.allFinite()) {
        points_global.push_back(landmark.p_FinG);
      }
    }
    for (const PersistentLandmark &landmark : landmarks_) {
      if (landmark.p_FinM.allFinite()) {
        points_map.push_back(landmark.p_FinM);
      }
    }

    double mean_error_m = -1.0;
    double max_error_m = -1.0;
    const float confidence = estimate_rigid_transform(points_global, points_map, T_M_G_estimate, mean_error_m, max_error_m);
    icp_succeeded = (confidence > 0.0f);
    if (icp_succeeded) {
      localized_pose_from_map = map_to_global_.transform * (T_M_G_estimate * current_pose_from_ekf);
      const Eigen::Isometry3d T_delta = T_M_G_estimate * T_M_G_initial.inverse();
      const double translation_delta = T_delta.translation().norm();
      const double rotation_delta = Eigen::AngleAxisd(T_delta.linear()).angle();
      localization_agrees_with_odom = translation_delta <= options_.pose_agreement_translation_m &&
                                      rotation_delta <= options_.pose_agreement_rotation_rad;
      map_to_global_.transform = T_M_G_estimate.inverse();
      map_to_global_.measurements++;
    } else {
      T_M_G_estimate = T_M_G_initial;
      localization_agrees_with_odom = false;
    }
  }

  match_landmarks(landmarks, T_M_G_estimate);

  if (landmarks_.size() < options_.min_map_landmarks_for_icp || localization_agrees_with_odom) {
    for (TrackedLandmark &landmark : landmarks) {
      if (landmark.landmark == nullptr || !landmark.landmark->should_marg || !landmark.p_FinM.allFinite()) {
        continue;
      }

      const size_t map_index = find_map_landmark_index(landmark.landmark->_map_landmark_id);
      if (map_index != INVALID_MAP_LANDMARK_ID) {
        PersistentLandmark &map_landmark = landmarks_.at(map_index);
        const double old_weight = static_cast<double>(std::max<size_t>(1, map_landmark.source_landmarks));
        map_landmark.p_FinM = (old_weight * map_landmark.p_FinM + landmark.p_FinM) / (old_weight + 1.0);
        map_landmark.source_landmarks++;
        map_landmark.match_hits++;
        merged_landmarks_++;
      } else {
        PersistentLandmark map_landmark;
        map_landmark.map_id = next_map_id_++;
        map_landmark.p_FinM = landmark.p_FinM;
        map_landmark.source_landmarks = 1;
        landmarks_.push_back(map_landmark);
        landmark.landmark->_map_landmark_id = map_landmark.map_id;
        spawned_landmarks_++;
      }
      processed_landmarks_++;
    }
  }

  if (!options_.update_state) {
    return current_pose_from_ekf;
  }

  return localized_pose_from_map;
}

bool UpdaterMap::match_landmarks(std::vector<TrackedLandmark> &landmarks, const Eigen::Isometry3d &T_M_G) {
  std::vector<bool> map_used(landmarks_.size(), false);
  bool matched_any = false;

  for (TrackedLandmark &landmark : landmarks) {
    if (landmark.landmark != nullptr) {
      landmark.landmark->_map_landmark_id = INVALID_MAP_LANDMARK_ID;
    }
    landmark.p_FinM = T_M_G * landmark.p_FinG;
    if (!landmark.p_FinM.allFinite()) {
      continue;
    }

    size_t best_index = INVALID_MAP_LANDMARK_ID;
    double best_distance = options_.spatial_match_radius_m;
    for (size_t i = 0; i < landmarks_.size(); ++i) {
      if (map_used.at(i)) {
        continue;
      }
      const double distance = (landmarks_.at(i).p_FinM - landmark.p_FinM).norm();
      if (distance < best_distance) {
        best_distance = distance;
        best_index = i;
      }
    }

    if (best_index == INVALID_MAP_LANDMARK_ID) {
      continue;
    }

    map_used.at(best_index) = true;
    landmarks_.at(best_index).match_attempts++;
    if (landmark.landmark != nullptr) {
      landmark.landmark->_map_landmark_id = landmarks_.at(best_index).map_id;
    }
    matched_any = true;
  }

  return matched_any;
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

double UpdaterMap::confidence_from_distance(double distance_m, double confident_radius_m, double max_radius_m) {
  if (max_radius_m <= confident_radius_m) {
    return (distance_m <= max_radius_m) ? 1.0 : 0.0;
  }
  if (distance_m <= confident_radius_m) {
    return 1.0;
  }
  if (distance_m >= max_radius_m) {
    return 0.0;
  }
  return 1.0 - (distance_m - confident_radius_m) / (max_radius_m - confident_radius_m);
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
