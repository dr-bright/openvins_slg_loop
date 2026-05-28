/*
 * openvins_lightglue: persistent SLG map accumulator.
 */

#ifndef OV_LIGHTGLUE_UPDATER_SLGM_H
#define OV_LIGHTGLUE_UPDATER_SLGM_H

#include "track/FeatureSLG.h"

#include <Eigen/Eigen>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

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

class UpdaterSLGM {
public:
  struct Options {
    double merge_distance_m = 0.15;
    size_t max_descriptors_per_landmark = 64;
  };

  UpdaterSLGM();

  explicit UpdaterSLGM(Options options);

  bool process_dying_landmark(size_t featid, const Eigen::Vector3d &p_FinG, const std::shared_ptr<FeatureSLG> &feature);

  const std::vector<SLGMLandmark> &landmarks() const { return landmarks_; }

  size_t map_size() const { return landmarks_.size(); }

  size_t processed_landmarks() const { return processed_landmarks_; }

  size_t spawned_landmarks() const { return spawned_landmarks_; }

  size_t merged_landmarks() const { return merged_landmarks_; }

  size_t stored_descriptors() const { return stored_descriptors_; }

protected:
  size_t find_spatial_match(const Eigen::Vector3d &p_FinG) const;

  std::vector<SLGMDescriptor> collect_descriptors(size_t featid, const std::shared_ptr<FeatureSLG> &feature) const;

  void append_descriptors(SLGMLandmark &landmark, const std::vector<SLGMDescriptor> &descriptors);

  void cull_descriptors(SLGMLandmark &landmark);

  Options options_;
  std::vector<SLGMLandmark> landmarks_;
  size_t next_map_id_ = 1;
  size_t processed_landmarks_ = 0;
  size_t spawned_landmarks_ = 0;
  size_t merged_landmarks_ = 0;
  size_t stored_descriptors_ = 0;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_UPDATER_SLGM_H
