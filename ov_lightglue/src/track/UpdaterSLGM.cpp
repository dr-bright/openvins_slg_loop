/*
 * openvins_lightglue: persistent SLG map accumulator.
 */

#include "track/UpdaterSLGM.h"

#include <algorithm>
#include <limits>

namespace ov_lightglue {

UpdaterSLGM::UpdaterSLGM() : options_(Options()) {}

UpdaterSLGM::UpdaterSLGM(Options options) : options_(options) {}

bool UpdaterSLGM::process_dying_landmark(size_t featid, const Eigen::Vector3d &p_FinG, const std::shared_ptr<FeatureSLG> &feature) {
  if (feature == nullptr || !p_FinG.allFinite()) {
    return false;
  }

  std::vector<SLGMDescriptor> descriptors = collect_descriptors(featid, feature);
  if (descriptors.empty()) {
    return false;
  }

  const size_t match_index = find_spatial_match(p_FinG);
  if (match_index < landmarks_.size()) {
    SLGMLandmark &landmark = landmarks_.at(match_index);
    const double old_weight = static_cast<double>(std::max<size_t>(1, landmark.source_landmarks));
    landmark.p_FinG = (old_weight * landmark.p_FinG + p_FinG) / (old_weight + 1.0);
    landmark.source_landmarks++;
    landmark.last_seen = descriptors.back().timestamp;
    append_descriptors(landmark, descriptors);
    merged_landmarks_++;
  } else {
    SLGMLandmark landmark;
    landmark.map_id = next_map_id_++;
    landmark.p_FinG = p_FinG;
    landmark.source_landmarks = 1;
    landmark.first_seen = descriptors.front().timestamp;
    landmark.last_seen = descriptors.back().timestamp;
    append_descriptors(landmark, descriptors);
    landmarks_.push_back(std::move(landmark));
    spawned_landmarks_++;
  }

  processed_landmarks_++;
  stored_descriptors_ = 0;
  for (const SLGMLandmark &landmark : landmarks_) {
    stored_descriptors_ += landmark.descriptors.size();
  }
  return true;
}

size_t UpdaterSLGM::find_spatial_match(const Eigen::Vector3d &p_FinG) const {
  size_t best_index = landmarks_.size();
  double best_distance = options_.merge_distance_m;
  for (size_t i = 0; i < landmarks_.size(); ++i) {
    const double distance = (landmarks_.at(i).p_FinG - p_FinG).norm();
    if (distance < best_distance) {
      best_distance = distance;
      best_index = i;
    }
  }
  return best_index;
}

std::vector<SLGMDescriptor> UpdaterSLGM::collect_descriptors(size_t featid, const std::shared_ptr<FeatureSLG> &feature) const {
  std::vector<SLGMDescriptor> descriptors;
  if (feature == nullptr) {
    return descriptors;
  }

  for (const auto &cam_descriptors : feature->descriptors) {
    const size_t cam_id = cam_descriptors.first;
    const std::vector<cv::Mat> &desc_vec = cam_descriptors.second;
    const auto timestamp_it = feature->descriptor_timestamps.find(cam_id);
    const auto score_it = feature->scores.find(cam_id);
    const auto uv_it = feature->uvs.find(cam_id);
    const auto uvn_it = feature->uvs_norm.find(cam_id);
    if (timestamp_it == feature->descriptor_timestamps.end()) {
      continue;
    }

    const std::vector<double> &timestamps = timestamp_it->second;
    const std::vector<float> *scores = (score_it != feature->scores.end()) ? &score_it->second : nullptr;
    const std::vector<Eigen::VectorXf> *uvs = (uv_it != feature->uvs.end()) ? &uv_it->second : nullptr;
    const std::vector<Eigen::VectorXf> *uvs_norm = (uvn_it != feature->uvs_norm.end()) ? &uvn_it->second : nullptr;
    const size_t count = std::min(desc_vec.size(), timestamps.size());
    for (size_t i = 0; i < count; ++i) {
      if (desc_vec.at(i).empty()) {
        continue;
      }
      SLGMDescriptor descriptor;
      descriptor.descriptor = desc_vec.at(i).clone();
      descriptor.source_feature_id = featid;
      descriptor.cam_id = cam_id;
      descriptor.timestamp = timestamps.at(i);
      descriptor.score = (scores != nullptr && i < scores->size()) ? scores->at(i) : 0.0f;
      descriptor.uv = (uvs != nullptr && i < uvs->size()) ? uvs->at(i) : Eigen::VectorXf();
      descriptor.uv_norm = (uvs_norm != nullptr && i < uvs_norm->size()) ? uvs_norm->at(i) : Eigen::VectorXf();
      descriptors.push_back(std::move(descriptor));
    }
  }

  std::sort(descriptors.begin(), descriptors.end(), [](const SLGMDescriptor &a, const SLGMDescriptor &b) {
    if (a.timestamp != b.timestamp) {
      return a.timestamp < b.timestamp;
    }
    return a.cam_id < b.cam_id;
  });
  return descriptors;
}

void UpdaterSLGM::append_descriptors(SLGMLandmark &landmark, const std::vector<SLGMDescriptor> &descriptors) {
  landmark.descriptors.insert(landmark.descriptors.end(), descriptors.begin(), descriptors.end());
  cull_descriptors(landmark);
}

void UpdaterSLGM::cull_descriptors(SLGMLandmark &landmark) {
  if (options_.max_descriptors_per_landmark == 0 || landmark.descriptors.size() <= options_.max_descriptors_per_landmark) {
    return;
  }

  std::sort(landmark.descriptors.begin(), landmark.descriptors.end(), [](const SLGMDescriptor &a, const SLGMDescriptor &b) {
    if (a.hits != b.hits) {
      return a.hits > b.hits;
    }
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.timestamp > b.timestamp;
  });
  landmark.descriptors.resize(options_.max_descriptors_per_landmark);
  std::sort(landmark.descriptors.begin(), landmark.descriptors.end(), [](const SLGMDescriptor &a, const SLGMDescriptor &b) {
    if (a.timestamp != b.timestamp) {
      return a.timestamp < b.timestamp;
    }
    return a.cam_id < b.cam_id;
  });
}

} // namespace ov_lightglue
