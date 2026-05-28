/*
 * openvins_lightglue: persistent SLG map accumulator.
 */

#ifndef OV_LIGHTGLUE_UPDATER_SLGM_H
#define OV_LIGHTGLUE_UPDATER_SLGM_H

#include "track/FeatureSLG.h"

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace ov_lightglue {

class TrackSLG;

} // namespace ov_lightglue

namespace ov_type {
class Landmark;
} // namespace ov_type

namespace ov_lightglue {

struct SLGMDescriptor {
  cv::Mat descriptor;
  size_t source_feature_id = 0;
  size_t cam_id = 0;
  Eigen::VectorXf uv;
  Eigen::VectorXf uv_norm;
  double timestamp = -1.0;
  float score = 0.0f;
  size_t hits = 0;
  size_t attempts = 0;
};

struct SLGMLandmark {
  size_t map_id = 0;
  Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
  std::vector<SLGMDescriptor> descriptors;
  size_t source_landmarks = 0;
  size_t match_hits = 0;
  size_t match_attempts = 0;
  double first_seen = -1.0;
  double last_seen = -1.0;
};

struct TrackedLandmarkSLGM {
  std::shared_ptr<ov_type::Landmark> landmark;
  size_t featid = 0;
  Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
  bool should_marg = false;
  int unique_camera_id = -1;
  int unobserved_count = 0;
  int update_fail_count = 0;
  size_t map_landmark_id = static_cast<size_t>(-1);
  double map_match_confidence = -1.0;
};

class UpdaterSLGM {
public:
  static constexpr size_t INVALID_MAP_LANDMARK_INDEX = static_cast<size_t>(-1);

  struct Options {
    size_t max_descriptors_per_landmark = 64;
  };

  explicit UpdaterSLGM(std::shared_ptr<TrackSLG> tracker = nullptr);

  explicit UpdaterSLGM(Options options, std::shared_ptr<TrackSLG> tracker = nullptr);

  void set_tracker(std::shared_ptr<TrackSLG> tracker) { tracker_ = std::move(tracker); }

  size_t process_landmarks(double timestamp, std::vector<TrackedLandmarkSLGM> landmarks);

  const std::vector<SLGMLandmark> &landmarks() const { return landmarks_; }

  const std::unordered_map<size_t, size_t> &ekf_map_assignments() const { return ekf_map_assignments_; }

  size_t map_size() const { return landmarks_.size(); }

  size_t processed_landmarks() const { return processed_landmarks_; }

  size_t spawned_landmarks() const { return spawned_landmarks_; }

  size_t merged_landmarks() const { return merged_landmarks_; }

  size_t stored_descriptors() const { return stored_descriptors_; }

protected:
  bool process_dying_landmark(const TrackedLandmarkSLGM &landmark, const std::shared_ptr<FeatureSLG> &feature);

  size_t process_dying_landmarks(const std::vector<TrackedLandmarkSLGM> &dying_landmarks);

  std::vector<TrackedLandmarkSLGM> match_current_landmarks(double timestamp, const std::vector<TrackedLandmarkSLGM> &current_landmarks);

  std::vector<SLGMDescriptor> collect_descriptors(size_t featid, const std::shared_ptr<FeatureSLG> &feature) const;

  void append_descriptors(SLGMLandmark &landmark, const std::vector<SLGMDescriptor> &descriptors);

  void cull_descriptors(SLGMLandmark &landmark);

  bool build_map_matching_set(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, std::vector<size_t> &landmark_indices) const;

  size_t find_map_landmark_index(size_t map_landmark_id) const;

  std::shared_ptr<FeatureSLG> get_feature_slg(size_t featid) const;

  bool get_current_observation(double timestamp, const std::shared_ptr<FeatureSLG> &feature, cv::KeyPoint &keypoint, cv::Mat &descriptor) const;

  Options options_;
  std::shared_ptr<TrackSLG> tracker_;
  std::vector<SLGMLandmark> landmarks_;
  std::unordered_map<size_t, size_t> ekf_map_assignments_;
  size_t next_map_id_ = 1;
  size_t processed_landmarks_ = 0;
  size_t spawned_landmarks_ = 0;
  size_t merged_landmarks_ = 0;
  size_t stored_descriptors_ = 0;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_UPDATER_SLGM_H
