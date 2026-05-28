/*
 * openvins_lightglue: FeatureDatabase with SuperPoint descriptor payloads.
 */

#ifndef OV_LIGHTGLUE_FEATURE_DATABASE_SLG_H
#define OV_LIGHTGLUE_FEATURE_DATABASE_SLG_H

#include "feat/FeatureDatabase.h"
#include "track/FeatureSLG.h"

#include <opencv2/core.hpp>

namespace ov_lightglue {

class FeatureDatabaseSLG : public ov_core::FeatureDatabase {
public:
  std::shared_ptr<FeatureSLG> get_feature_slg(size_t id, bool remove = false);

  void update_feature_slg(size_t id, double timestamp, size_t cam_id, float u, float v, float u_n, float v_n,
                          const cv::Mat &descriptor, float score);
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_FEATURE_DATABASE_SLG_H
