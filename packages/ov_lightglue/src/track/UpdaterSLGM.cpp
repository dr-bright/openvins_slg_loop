/*
 * openvins_lightglue: persistent SLG map accumulator.
 */

#include "track/UpdaterSLGM.h"

#include "track/FeatureDatabaseSLG.h"
#include "track/TrackSLG.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ov_lightglue {

UpdaterSLGM::UpdaterSLGM(std::shared_ptr<TrackSLG> tracker) : options_(Options()), tracker_(std::move(tracker)) {}

UpdaterSLGM::UpdaterSLGM(Options options, std::shared_ptr<TrackSLG> tracker) : options_(options), tracker_(std::move(tracker)) {}

size_t UpdaterSLGM::process_landmarks(double timestamp, std::vector<TrackedLandmarkSLGM> tracked_landmarks) {
  match_current_landmarks(timestamp, tracked_landmarks);

  std::unordered_map<size_t, TrackedLandmarkSLGM*> assignments_by_featid;
  assignments_by_featid.reserve(assignments.size());
  for (const TrackedLandmarkSLGM &assignment : tracked_landmarks) {
    assignments_by_featid[assignment.featid] = &assignment;
  }

  std::vector<TrackedLandmarkSLGM> dying_landmarks;
  dying_landmarks.reserve(tracked_landmarks.size());
  for (TrackedLandmarkSLGM &tracked_landmark : tracked_landmarks) {
    const auto assignment = assignments_by_featid.find(tracked_landmark.featid);
    if (assignment != assignments_by_featid.end()) {
      tracked_landmark.map_landmark_id = assignment->second->map_landmark_id;
      tracked_landmark.map_match_confidence = assignment->second->map_match_confidence;
    }
    if (tracked_landmark.should_marg) {
      dying_landmarks.push_back(tracked_landmark);
    }
  }

  return process_dying_landmarks(dying_landmarks);
}

bool UpdaterSLGM::process_dying_landmark(const TrackedLandmarkSLGM &tracked_landmark, const std::shared_ptr<FeatureSLG> &feature) {
  if (feature == nullptr || !tracked_landmark.p_FinG.allFinite()) {
    return false;
  }

  std::vector<SLGMDescriptor> descriptors = collect_descriptors(tracked_landmark.featid, feature);
  if (descriptors.empty()) {
    return false;
  }

  const size_t map_landmark_index = find_map_landmark_index(tracked_landmark.map_landmark_id);
  if (map_landmark_index != INVALID_MAP_LANDMARK_INDEX && map_landmark_index < landmarks_.size()) {
    SLGMLandmark &landmark = landmarks_.at(map_landmark_index);
    const double old_weight = static_cast<double>(std::max<size_t>(1, landmark.source_landmarks));
    landmark.p_FinG = (old_weight * landmark.p_FinG + tracked_landmark.p_FinG) / (old_weight + 1.0);
    landmark.source_landmarks++;
    landmark.last_seen = descriptors.back().timestamp;
    append_descriptors(landmark, descriptors);
    merged_landmarks_++;
  } else {
    SLGMLandmark landmark;
    landmark.map_id = next_map_id_++;
    landmark.p_FinG = tracked_landmark.p_FinG;
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

std::vector<SLGMDescriptor> UpdaterSLGM::collect_descriptors(size_t featid, const std::shared_ptr<FeatureSLG> &feature) const {
  std::vector<SLGMDescriptor> descriptors;
  if (feature == nullptr) {
    return descriptors;
  }

  for (const auto &cam_descriptors : feature->descriptors) {
    const size_t cam_id = cam_descriptors.first;
    const std::vector<cv::Mat> &desc_vec = cam_descriptors.second;
    const auto timestamp_it = feature->timestamps.find(cam_id);
    const auto score_it = feature->scores.find(cam_id);
    const auto uv_it = feature->uvs.find(cam_id);
    const auto uvn_it = feature->uvs_norm.find(cam_id);
    if (timestamp_it == feature->timestamps.end()) {
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

size_t UpdaterSLGM::process_dying_landmarks(const std::vector<TrackedLandmarkSLGM> &dying_landmarks) {
  size_t processed = 0;
  for (TrackedLandmarkSLGM dying_landmark : dying_landmarks) {
    const auto assignment = ekf_map_assignments_.find(dying_landmark.featid);
    if (dying_landmark.map_landmark_id == INVALID_MAP_LANDMARK_INDEX && assignment != ekf_map_assignments_.end()) {
      dying_landmark.map_landmark_id = assignment->second;
    }
    if (process_dying_landmark(dying_landmark, get_feature_slg(dying_landmark.featid))) {
      processed++;
    }
    ekf_map_assignments_.erase(dying_landmark.featid);
  }
  return processed;
}

void UpdaterSLGM::match_current_landmarks(
  double timestamp,
  std::vector<TrackedLandmarkSLGM> &current_landmarks
) {
  if (tracker_ == nullptr || current_landmarks.empty() || landmarks_.empty()) {
    return;
  }

  std::vector<cv::KeyPoint> map_keypoints;
  cv::Mat map_descriptors;
  std::vector<size_t> map_indices;
  if (!build_map_matching_set(map_keypoints, map_descriptors, map_indices)) {
    return;
  }

  std::vector<cv::KeyPoint> current_keypoints;
  cv::Mat current_descriptors;
  std::vector<TrackedLandmarkSLGM*> current_tracked_landmarks;
  current_keypoints.reserve(current_landmarks.size());
  current_tracked_landmarks.reserve(current_landmarks.size());
  for (auto &current_landmark : current_landmarks) {
    cv::KeyPoint keypoint;
    cv::Mat descriptor;
    if (!get_current_observation(timestamp, get_feature_slg(current_landmark.featid), keypoint, descriptor)) {
      continue;
    }
    current_keypoints.push_back(keypoint);
    current_tracked_landmarks.push_back(&current_landmark);
    TrackSLG::append_descriptor_row(descriptor, 0, current_descriptors);
  }

  if (current_keypoints.empty() || current_descriptors.empty()) {
    return;
  }

  std::vector<cv::DMatch> matches;
  tracker_->run_lightglue(cv::Size(0, 0), map_keypoints, map_descriptors, cv::Size(0, 0), current_keypoints, current_descriptors, matches, false);

  for (const cv::DMatch &match : matches) {
    if (match.queryIdx < 0 || match.trainIdx < 0 || match.queryIdx >= static_cast<int>(map_indices.size()) ||
        match.trainIdx >= static_cast<int>(current_tracked_landmarks.size())) {
      continue;
    }
    const size_t map_landmark_index = map_indices.at(static_cast<size_t>(match.queryIdx));
    SLGMLandmark &map_landmark = landmarks_.at(map_landmark_index);
    map_landmark.match_hits++;
    map_landmark.last_seen = timestamp;

    TrackedLandmarkSLGM* assignment = current_tracked_landmarks.at(static_cast<size_t>(match.trainIdx));
    assignment->map_landmark_id = map_landmark.map_id;
    assignment->map_match_confidence = match.distance;
    ekf_map_assignments_[assignment->featid] = assignment->map_landmark_id;
  }
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

bool UpdaterSLGM::build_map_matching_set(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, std::vector<size_t> &landmark_indices) const {
  keypoints.clear();
  descriptors.release();
  landmark_indices.clear();

  for (size_t landmark_index = 0; landmark_index < landmarks_.size(); ++landmark_index) {
    const SLGMLandmark &landmark = landmarks_.at(landmark_index);
    for (const SLGMDescriptor &descriptor : landmark.descriptors) {
      if (descriptor.descriptor.empty()) {
        continue;
      }
      cv::KeyPoint keypoint;
      keypoint.pt.x = static_cast<float>(landmark.p_FinG.x());
      keypoint.pt.y = static_cast<float>(landmark.p_FinG.y());
      keypoint.response = descriptor.score;
      keypoints.push_back(keypoint);
      landmark_indices.push_back(landmark_index);
      TrackSLG::append_descriptor_row(descriptor.descriptor, 0, descriptors);
    }
  }

  return !keypoints.empty() && !descriptors.empty();
}

size_t UpdaterSLGM::find_map_landmark_index(size_t map_landmark_id) const {
  if (map_landmark_id == INVALID_MAP_LANDMARK_INDEX) {
    return INVALID_MAP_LANDMARK_INDEX;
  }
  for (size_t i = 0; i < landmarks_.size(); ++i) {
    if (landmarks_.at(i).map_id == map_landmark_id) {
      return i;
    }
  }
  return INVALID_MAP_LANDMARK_INDEX;
}

std::shared_ptr<FeatureSLG> UpdaterSLGM::get_feature_slg(size_t featid) const {
  if (tracker_ == nullptr || tracker_->get_feature_database() == nullptr) {
    return nullptr;
  }
  std::shared_ptr<FeatureDatabaseSLG> database_slg = std::static_pointer_cast<FeatureDatabaseSLG>(tracker_->get_feature_database());
  if (database_slg == nullptr) {
    return nullptr;
  }
  return database_slg->get_feature_slg(featid, false);
}

bool UpdaterSLGM::get_current_observation(double timestamp, const std::shared_ptr<FeatureSLG> &feature, cv::KeyPoint &keypoint,
                                          cv::Mat &descriptor) const {
  if (feature == nullptr || feature->to_delete) {
    return false;
  }

  for (const auto &cam_descriptors : feature->descriptors) {
    const size_t cam_id = cam_descriptors.first;
    const std::vector<cv::Mat> &descriptors = cam_descriptors.second;
    const auto timestamp_it = feature->timestamps.find(cam_id);
    const auto uv_it = feature->uvs.find(cam_id);
    if (timestamp_it == feature->timestamps.end() || uv_it == feature->uvs.end()) {
      continue;
    }
    const std::vector<double> &timestamps = timestamp_it->second;
    const std::vector<Eigen::VectorXf> &uvs = uv_it->second;
    const size_t count = std::min(descriptors.size(), std::min(timestamps.size(), uvs.size()));
    for (size_t i = 0; i < count; ++i) {
      if (std::abs(timestamps.at(i) - timestamp) > 1e-9 || descriptors.at(i).empty() || uvs.at(i).rows() < 2) {
        continue;
      }
      keypoint = cv::KeyPoint();
      keypoint.pt.x = uvs.at(i)(0);
      keypoint.pt.y = uvs.at(i)(1);
      descriptor = descriptors.at(i).clone();
      return true;
    }
  }
  return false;
}

} // namespace ov_lightglue
