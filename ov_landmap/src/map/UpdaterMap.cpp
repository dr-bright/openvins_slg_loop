/*
 * ov_landmap: descriptor-free persistent landmark map.
 */

#include "map/UpdaterMap.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ov_landmap {

UpdaterMap::UpdaterMap() : options_(Options()) {}

UpdaterMap::UpdaterMap(Options options) : options_(options) {}

PoseEstimateMap UpdaterMap::process_landmarks(const std::vector<TrackedLandmarkMap> &landmarks,
                                              const Eigen::Isometry3d &current_pose_from_ekf) {
  PoseEstimateMap pose_estimate = estimate_pose(landmarks, current_pose_from_ekf);
  const Eigen::Isometry3d T_M_E = pose_estimate.valid ? pose_estimate.T_M_E : Eigen::Isometry3d::Identity();
  std::vector<TrackedLandmarkMap> matched_landmarks = match_landmarks(landmarks, T_M_E);

  std::vector<TrackedLandmarkMap> dying_landmarks;
  dying_landmarks.reserve(matched_landmarks.size());
  for (const TrackedLandmarkMap &landmark : matched_landmarks) {
    if (landmark.should_marg) {
      dying_landmarks.push_back(landmark);
    }
  }
  process_dying_landmarks(dying_landmarks);
  return pose_estimate;
}

PoseEstimateMap UpdaterMap::estimate_pose(const std::vector<TrackedLandmarkMap> &landmarks,
                                          const Eigen::Isometry3d &current_pose_from_ekf) const {
  PoseEstimateMap estimate;
  estimate.current_pose_from_ekf = current_pose_from_ekf;
  estimate.T_M_E = Eigen::Isometry3d::Identity();

  const std::vector<TrackedLandmarkMap> matched_landmarks = match_landmarks(landmarks, Eigen::Isometry3d::Identity());
  std::vector<Eigen::Vector3d> points_ekf;
  std::vector<Eigen::Vector3d> points_map;
  points_ekf.reserve(matched_landmarks.size());
  points_map.reserve(matched_landmarks.size());

  for (const TrackedLandmarkMap &landmark : matched_landmarks) {
    const size_t map_index = find_map_landmark_index(landmark.map_landmark_id);
    if (map_index == INVALID_MAP_LANDMARK_ID) {
      continue;
    }
    points_ekf.push_back(landmark.p_FinG);
    points_map.push_back(landmarks_.at(map_index).p_FinM);
  }

  estimate.num_matches = points_ekf.size();
  if (points_ekf.size() < options_.min_pose_matches) {
    return estimate;
  }

  estimate.valid = estimate_rigid_transform(points_ekf, points_map, estimate.T_M_E, estimate.mean_match_error_m, estimate.max_match_error_m);
  return estimate;
}

bool UpdaterMap::update_state_with_pose_estimate(const PoseEstimateMap &pose_estimate) {
  return pose_estimate.valid;
}

std::vector<TrackedLandmarkMap> UpdaterMap::match_landmarks(const std::vector<TrackedLandmarkMap> &landmarks,
                                                           const Eigen::Isometry3d &T_M_E) const {
  std::vector<TrackedLandmarkMap> matched_landmarks = landmarks;
  std::vector<bool> map_used(landmarks_.size(), false);

  for (TrackedLandmarkMap &landmark : matched_landmarks) {
    landmark.map_landmark_id = INVALID_MAP_LANDMARK_ID;
    landmark.map_match_distance = -1.0;
    landmark.map_match_confidence = -1.0;
    if (!landmark.p_FinG.allFinite()) {
      continue;
    }

    const Eigen::Vector3d p_FinM = T_M_E * landmark.p_FinG;
    landmark.p_FinM = p_FinM;
    landmark.has_p_FinM = true;
    size_t best_index = INVALID_MAP_LANDMARK_ID;
    double best_distance = options_.spatial_match_radius_m;
    for (size_t i = 0; i < landmarks_.size(); ++i) {
      if (map_used.at(i)) {
        continue;
      }
      const double distance = (landmarks_.at(i).p_FinM - p_FinM).norm();
      if (distance < best_distance) {
        best_distance = distance;
        best_index = i;
      }
    }

    if (best_index == INVALID_MAP_LANDMARK_ID) {
      continue;
    }
    map_used.at(best_index) = true;
    landmark.map_landmark_id = landmarks_.at(best_index).map_id;
    landmark.map_match_distance = best_distance;
    landmark.map_match_confidence =
        confidence_from_distance(best_distance, options_.confident_match_radius_m, options_.spatial_match_radius_m);
  }

  return matched_landmarks;
}

size_t UpdaterMap::process_dying_landmarks(const std::vector<TrackedLandmarkMap> &dying_landmarks) {
  size_t processed = 0;
  for (TrackedLandmarkMap landmark : dying_landmarks) {
    const auto assignment = ekf_map_assignments_.find(landmark.featid);
    if (landmark.map_landmark_id == INVALID_MAP_LANDMARK_ID && assignment != ekf_map_assignments_.end()) {
      landmark.map_landmark_id = assignment->second;
    }
    if (process_dying_landmark(landmark)) {
      processed++;
    }
    ekf_map_assignments_.erase(landmark.featid);
  }
  return processed;
}

bool UpdaterMap::process_dying_landmark(const TrackedLandmarkMap &landmark) {
  const Eigen::Vector3d p_FinM = landmark.has_p_FinM ? landmark.p_FinM : landmark.p_FinG;
  if (!p_FinM.allFinite()) {
    return false;
  }

  const size_t map_index = find_map_landmark_index(landmark.map_landmark_id);
  if (map_index != INVALID_MAP_LANDMARK_ID && map_index < landmarks_.size()) {
    PersistentLandmarkMap &map_landmark = landmarks_.at(map_index);
    const double old_weight = static_cast<double>(std::max<size_t>(1, map_landmark.source_landmarks));
    map_landmark.p_FinM = (old_weight * map_landmark.p_FinM + p_FinM) / (old_weight + 1.0);
    map_landmark.source_landmarks++;
    map_landmark.match_hits++;
    merged_landmarks_++;
  } else {
    PersistentLandmarkMap map_landmark;
    map_landmark.map_id = next_map_id_++;
    map_landmark.p_FinM = p_FinM;
    map_landmark.source_landmarks = 1;
    landmarks_.push_back(map_landmark);
    spawned_landmarks_++;
  }

  processed_landmarks_++;
  return true;
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

bool UpdaterMap::estimate_rigid_transform(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &dst,
                                          Eigen::Isometry3d &T_dst_src, double &mean_error_m, double &max_error_m) {
  if (src.size() != dst.size() || src.size() < 3) {
    return false;
  }

  Eigen::Vector3d mean_src = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_dst = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    mean_src += src.at(i);
    mean_dst += dst.at(i);
  }
  mean_src /= static_cast<double>(src.size());
  mean_dst /= static_cast<double>(dst.size());

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    H += (src.at(i) - mean_src) * (dst.at(i) - mean_dst).transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (svd.matrixU().cols() != 3 || svd.matrixV().cols() != 3) {
    return false;
  }

  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
  if (R.determinant() < 0.0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1.0;
    R = V * svd.matrixU().transpose();
  }
  const Eigen::Vector3d t = mean_dst - R * mean_src;

  T_dst_src = Eigen::Isometry3d::Identity();
  T_dst_src.linear() = R;
  T_dst_src.translation() = t;

  mean_error_m = 0.0;
  max_error_m = 0.0;
  for (size_t i = 0; i < src.size(); ++i) {
    const double error = (T_dst_src * src.at(i) - dst.at(i)).norm();
    mean_error_m += error;
    max_error_m = std::max(max_error_m, error);
  }
  mean_error_m /= static_cast<double>(src.size());
  return std::isfinite(mean_error_m) && std::isfinite(max_error_m);
}

} // namespace ov_landmap
