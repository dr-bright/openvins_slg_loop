/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#include "track/FeatureDatabaseDINO.h"

#include <Eigen/Eigen>
#include <mutex>

namespace ov_dino {

void FeatureDatabaseDINO::update_feature_dino(size_t id, double timestamp, size_t cam_id, float u, float v, float u_n, float v_n,
                                              const FeatureDINOMeta &meta) {
  std::lock_guard<std::mutex> lck(mtx);
  std::shared_ptr<FeatureDINO> feat;

  const auto it = features_idlookup.find(id);
  if (it != features_idlookup.end()) {
    feat = std::dynamic_pointer_cast<FeatureDINO>(it->second);
    if (feat == nullptr) {
      feat = std::make_shared<FeatureDINO>();
      feat->featid = it->second->featid;
      feat->to_delete = it->second->to_delete;
      feat->uvs = it->second->uvs;
      feat->uvs_norm = it->second->uvs_norm;
      feat->timestamps = it->second->timestamps;
      feat->anchor_cam_id = it->second->anchor_cam_id;
      feat->anchor_clone_timestamp = it->second->anchor_clone_timestamp;
      feat->p_FinA = it->second->p_FinA;
      feat->p_FinG = it->second->p_FinG;
      it->second = feat;
    }
  } else {
    feat = std::make_shared<FeatureDINO>();
    feat->featid = id;
    features_idlookup[id] = feat;
  }

  feat->uvs[cam_id].push_back(Eigen::Vector2f(u, v));
  feat->uvs_norm[cam_id].push_back(Eigen::Vector2f(u_n, v_n));
  feat->timestamps[cam_id].push_back(timestamp);
  feat->dino_meta[cam_id].push_back(meta);
}

std::shared_ptr<FeatureDINO> FeatureDatabaseDINO::get_feature_dino(size_t id, bool remove) {
  return std::dynamic_pointer_cast<FeatureDINO>(get_feature(id, remove));
}

} // namespace ov_dino
