/*
 * ov_landmap: descriptor-free persistent landmark map.
 */

#ifndef OV_LANDMAP_UPDATERMAP_H
#define OV_LANDMAP_UPDATERMAP_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ov_type {
class Landmark;
} // namespace ov_type

namespace ov_landmap {

struct TrackedLandmarkMap {
  std::shared_ptr<ov_type::Landmark> landmark;
  size_t featid = 0;
  Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_FinM = Eigen::Vector3d::Zero();
  bool has_p_FinM = false;
  bool should_marg = false;
  int unique_camera_id = -1;
  int unobserved_count = 0;
  int update_fail_count = 0;
  size_t map_landmark_id = static_cast<size_t>(-1);
  double map_match_distance = -1.0;
  double map_match_confidence = -1.0;
};

struct PersistentLandmarkMap {
  size_t map_id = 0;
  Eigen::Vector3d p_FinM = Eigen::Vector3d::Zero();
  size_t source_landmarks = 0;
  size_t match_hits = 0;
  size_t match_attempts = 0;
  double first_seen = -1.0;
  double last_seen = -1.0;
};

struct PoseEstimateMap {
  bool valid = false;
  Eigen::Isometry3d T_M_E = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d current_pose_from_ekf = Eigen::Isometry3d::Identity();
  size_t num_matches = 0;
  double mean_match_error_m = -1.0;
  double max_match_error_m = -1.0;
};

class UpdaterMap {
public:
  static constexpr size_t INVALID_MAP_LANDMARK_ID = static_cast<size_t>(-1);

  struct Options {
    double spatial_match_radius_m = 0.05;
    double confident_match_radius_m = 0.05;
    size_t min_pose_matches = 3;
  };

  UpdaterMap();

  explicit UpdaterMap(Options options);

  PoseEstimateMap process_landmarks(const std::vector<TrackedLandmarkMap> &landmarks,
                                    const Eigen::Isometry3d &current_pose_from_ekf);

  PoseEstimateMap estimate_pose(const std::vector<TrackedLandmarkMap> &landmarks,
                                const Eigen::Isometry3d &current_pose_from_ekf) const;

  bool update_state_with_pose_estimate(const PoseEstimateMap &pose_estimate);

  const std::vector<PersistentLandmarkMap> &landmarks() const { return landmarks_; }

  size_t map_size() const { return landmarks_.size(); }

  size_t processed_landmarks() const { return processed_landmarks_; }

  size_t spawned_landmarks() const { return spawned_landmarks_; }

  size_t merged_landmarks() const { return merged_landmarks_; }

protected:
  std::vector<TrackedLandmarkMap> match_landmarks(const std::vector<TrackedLandmarkMap> &landmarks,
                                                  const Eigen::Isometry3d &T_M_E) const;

  size_t process_dying_landmarks(const std::vector<TrackedLandmarkMap> &dying_landmarks);

  bool process_dying_landmark(const TrackedLandmarkMap &landmark);

  size_t find_map_landmark_index(size_t map_landmark_id) const;

  static double confidence_from_distance(double distance_m, double confident_radius_m, double max_radius_m);

  static bool estimate_rigid_transform(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &dst,
                                       Eigen::Isometry3d &T_dst_src, double &mean_error_m, double &max_error_m);

  Options options_;
  std::vector<PersistentLandmarkMap> landmarks_;
  std::unordered_map<size_t, size_t> ekf_map_assignments_;
  size_t next_map_id_ = 1;
  size_t processed_landmarks_ = 0;
  size_t spawned_landmarks_ = 0;
  size_t merged_landmarks_ = 0;
};

} // namespace ov_landmap

#endif // OV_LANDMAP_UPDATERMAP_H
