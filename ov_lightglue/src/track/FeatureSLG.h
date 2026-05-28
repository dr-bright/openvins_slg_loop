/*
 * openvins_lightglue: SLG feature metadata.
 */

#ifndef OV_LIGHTGLUE_FEATURE_SLG_H
#define OV_LIGHTGLUE_FEATURE_SLG_H

#include "feat/Feature.h"

#include <opencv2/core.hpp>

namespace ov_lightglue {

class FeatureSLG : public ov_core::Feature {
public:
  /// SuperPoint descriptors by camera ID.
  std::unordered_map<size_t, std::vector<cv::Mat>> descriptors;

  /// Timestamps aligned with descriptors by camera ID.
  std::unordered_map<size_t, std::vector<double>> descriptor_timestamps;

  /// SuperPoint detector scores aligned with descriptors by camera ID.
  std::unordered_map<size_t, std::vector<float>> scores;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_FEATURE_SLG_H
