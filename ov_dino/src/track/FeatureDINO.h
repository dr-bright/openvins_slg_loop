/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#ifndef OV_DINO_FEATURE_DINO_H
#define OV_DINO_FEATURE_DINO_H

#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "feat/Feature.h"

namespace ov_dino {

struct FeatureDINOMeta {
  cv::Rect2f bbox;
  float confidence = 0.0f;
  std::string label;
};

class FeatureDINO : public ov_core::Feature {
public:
  ~FeatureDINO() override = default;

  /// DINO metadata aligned with Feature::timestamps/uvs by camera ID.
  std::unordered_map<size_t, std::vector<FeatureDINOMeta>> dino_meta;
};

} // namespace ov_dino

#endif // OV_DINO_FEATURE_DINO_H
