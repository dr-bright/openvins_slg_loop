/*
 * openvins_lightglue: FeatureDatabase with SuperPoint descriptor payloads.
 */

#include "track/FeatureDatabaseSLG.h"

namespace ov_lightglue {

std::shared_ptr<FeatureSLG> FeatureDatabaseSLG::get_feature_slg(size_t id, bool remove) {
  return std::dynamic_pointer_cast<FeatureSLG>(get_feature(id, remove));
}

void FeatureDatabaseSLG::update_feature_slg(size_t id, double timestamp, size_t cam_id, float u, float v, float u_n, float v_n,
                                            const cv::Mat &descriptor, float score) {
  std::lock_guard<std::mutex> lck(mtx);

  std::shared_ptr<FeatureSLG> feat;
  const auto it = features_idlookup.find(id);
  if (it != features_idlookup.end()) {
    feat = std::dynamic_pointer_cast<FeatureSLG>(it->second);
    if (feat == nullptr) {
      feat = std::make_shared<FeatureSLG>();
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
    feat = std::make_shared<FeatureSLG>();
    feat->featid = id;
    features_idlookup[id] = feat;
  }

  if (feat->to_delete) {
    feat->uvs.clear();
    feat->uvs_norm.clear();
    feat->timestamps.clear();
    feat->descriptors.clear();
    feat->descriptor_timestamps.clear();
    feat->scores.clear();
    feat->to_delete = false;
  }

  feat->uvs[cam_id].push_back(Eigen::Vector2f(u, v));
  feat->uvs_norm[cam_id].push_back(Eigen::Vector2f(u_n, v_n));
  feat->timestamps[cam_id].push_back(timestamp);
  feat->descriptors[cam_id].push_back(descriptor.clone());
  feat->descriptor_timestamps[cam_id].push_back(timestamp);
  feat->scores[cam_id].push_back(score);
}

} // namespace ov_lightglue
