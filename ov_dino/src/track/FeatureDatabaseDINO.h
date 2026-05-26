/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#ifndef OV_DINO_FEATURE_DATABASE_DINO_H
#define OV_DINO_FEATURE_DATABASE_DINO_H

#include <memory>

#include "feat/FeatureDatabase.h"
#include "track/FeatureDINO.h"

namespace ov_dino {

class FeatureDatabaseDINO : public ov_core::FeatureDatabase {
public:
  void update_feature_dino(size_t id, double timestamp, size_t cam_id, float u, float v, float u_n, float v_n,
                           const FeatureDINOMeta &meta);

  std::shared_ptr<FeatureDINO> get_feature_dino(size_t id, bool remove = false);
};

} // namespace ov_dino

#endif // OV_DINO_FEATURE_DATABASE_DINO_H
