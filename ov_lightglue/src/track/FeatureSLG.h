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
  void export_latest_metadata(std::map<std::string, uint64_t> &u64_fields, std::map<std::string, int64_t> &i64_fields,
                              std::map<std::string, double> &f64_fields) const override;

  /// SuperPoint descriptors by camera ID.
  std::unordered_map<size_t, std::vector<cv::Mat>> descriptors;

  /// SuperPoint detector scores aligned with descriptors by camera ID.
  std::unordered_map<size_t, std::vector<float>> scores;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_FEATURE_SLG_H
