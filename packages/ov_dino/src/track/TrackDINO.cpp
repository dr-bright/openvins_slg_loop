/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#include "track/TrackDINO.h"

#include "cam/CamBase.h"
#include "track/FeatureDINO.h"
#include "track/FeatureDatabaseDINO.h"
#include "utils/print.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace ov_dino {
namespace {

constexpr float DINO_MATCH_MAX_BOOTSTRAP_CENTER_DIST_PX = 140.0f;
constexpr float DINO_MATCH_MAX_PREDICTED_CENTER_DIST_PX = 90.0f;
constexpr float DINO_MATCH_CENTER_GATE_BBOX_DIAG_MULT = 1.5f;
constexpr float DINO_MATCH_BOOTSTRAP_CENTER_GATE_BBOX_DIAG_MULT = 2.5f;
constexpr float DINO_MATCH_MAX_SIZE_LOG_RATIO = 0.85f;
constexpr float DINO_MATCH_BOOTSTRAP_MAX_SIZE_LOG_RATIO = 1.60f;
constexpr float DINO_MATCH_CENTER_COST_WEIGHT = 1.0f;
constexpr float DINO_MATCH_SIZE_COST_WEIGHT = 0.35f;
constexpr float DINO_MATCH_IOU_COST_WEIGHT = 0.60f;
constexpr float DINO_MATCH_ACCEPT_COST = 2.20f;

std::string normalize_label(const std::string &label) {
  size_t first = 0;
  while (first < label.size() && std::isspace(static_cast<unsigned char>(label.at(first)))) {
    ++first;
  }
  size_t last = label.size();
  while (last > first) {
    const char c = label.at(last - 1);
    if (!std::isspace(static_cast<unsigned char>(c)) && c != '.') {
      break;
    }
    --last;
  }

  std::string normalized;
  normalized.reserve(last - first);
  for (size_t i = first; i < last; ++i) {
    normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(label.at(i)))));
  }
  return normalized;
}

std::map<std::string, size_t> build_class_lookup(const std::vector<std::string> &classes) {
  std::map<std::string, size_t> lookup;
  for (size_t i = 0; i < classes.size(); ++i) {
    const std::string key = normalize_label(classes.at(i));
    if (!key.empty() && lookup.find(key) == lookup.end()) {
      lookup[key] = i;
    }
  }
  return lookup;
}

void collect_existing_ids(const std::shared_ptr<ov_core::FeatureDatabase> &database, size_t class_count, std::set<size_t> &used_feature_ids,
                          std::map<size_t, std::set<size_t>> &used_inclass_ids_by_class) {
  if (database == nullptr || class_count == 0) {
    return;
  }

  const auto features = database->get_internal_data();
  for (const auto &entry : features) {
    used_feature_ids.insert(entry.first);
    const size_t class_id = entry.first % class_count;
    used_inclass_ids_by_class[class_id].insert(entry.first / class_count);
  }
}

bool allocate_dino_feature_id(size_t class_id, size_t class_count, std::set<size_t> &used_feature_ids,
                              std::map<size_t, std::set<size_t>> &used_inclass_ids_by_class, size_t &feature_id, size_t &inclass_id) {
  if (class_count == 0 || class_id >= class_count) {
    return false;
  }

  std::set<size_t> &used_inclass_ids = used_inclass_ids_by_class[class_id];
  for (size_t candidate = 0; candidate < std::numeric_limits<size_t>::max() / class_count; ++candidate) {
    if (used_inclass_ids.find(candidate) != used_inclass_ids.end()) {
      continue;
    }
    const size_t candidate_feature_id = candidate * class_count + class_id;
    if (used_feature_ids.find(candidate_feature_id) != used_feature_ids.end()) {
      continue;
    }

    used_inclass_ids.insert(candidate);
    used_feature_ids.insert(candidate_feature_id);
    inclass_id = candidate;
    feature_id = candidate_feature_id;
    return true;
  }

  return false;
}

cv::Point2f bbox_center(const cv::Rect2f &bbox) {
  return cv::Point2f(bbox.x + 0.5f * bbox.width, bbox.y + 0.5f * bbox.height);
}

float rect_diag(const cv::Rect2f &bbox) {
  return std::sqrt(bbox.width * bbox.width + bbox.height * bbox.height);
}

float point_dist(const cv::Point2f &a, const cv::Point2f &b) {
  const cv::Point2f d = a - b;
  return std::sqrt(d.x * d.x + d.y * d.y);
}

float bbox_iou(const cv::Rect2f &a, const cv::Rect2f &b) {
  const float x0 = std::max(a.x, b.x);
  const float y0 = std::max(a.y, b.y);
  const float x1 = std::min(a.x + a.width, b.x + b.width);
  const float y1 = std::min(a.y + a.height, b.y + b.height);
  const float intersection = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
  const float union_area = a.area() + b.area() - intersection;
  return union_area > 1e-6f ? intersection / union_area : 0.0f;
}

float size_log_ratio(const cv::Rect2f &a, const cv::Rect2f &b) {
  const float area_a = std::max(1.0f, a.area());
  const float area_b = std::max(1.0f, b.area());
  return std::abs(std::log(area_a / area_b));
}

} // namespace

std::vector<TrackDINO::DinoTrackCandidate> TrackDINO::perform_matching_standalone(const std::vector<DinoDetection> &detections, size_t cam_id,
                                                                                  double timestamp) const {
  std::vector<DinoTrackCandidate> candidates;
  if (config_.prompts.empty()) {
    return candidates;
  }

  struct TrackState {
    size_t feature_id = 0;
    size_t class_id = 0;
    size_t inclass_id = 0;
    cv::Rect2f last_bbox;
    cv::Point2f predicted_center;
    bool has_velocity = false;
  };

  struct MatchEdge {
    size_t detection_index = 0;
    size_t track_index = 0;
    float cost = 0.0f;
  };

  const size_t class_count = config_.prompts.size();
  const std::map<std::string, size_t> class_lookup = build_class_lookup(config_.prompts);
  std::set<size_t> used_feature_ids;
  std::map<size_t, std::set<size_t>> used_inclass_ids_by_class;
  collect_existing_ids(database, class_count, used_feature_ids, used_inclass_ids_by_class);

  std::vector<DinoTrackCandidate> detection_candidates;
  detection_candidates.reserve(detections.size());
  for (const DinoDetection &detection : detections) {
    const auto class_it = class_lookup.find(normalize_label(detection.label));
    if (class_it == class_lookup.end()) {
      PRINT_DEBUG("[DINO]: skipping detection with unknown class label '%s'\n", detection.label.c_str());
      continue;
    }
    DinoTrackCandidate candidate;
    candidate.detection = detection;
    candidate.class_id = class_it->second;
    detection_candidates.push_back(candidate);
  }

  std::vector<TrackState> tracks;
  const auto features = database->get_internal_data();
  tracks.reserve(features.size());
  for (const auto &entry : features) {
    const size_t feature_id = entry.first;
    const size_t class_id = feature_id % class_count;
    const std::shared_ptr<FeatureDINO> feature = std::dynamic_pointer_cast<FeatureDINO>(entry.second);
    if (feature == nullptr || feature->dino_meta.count(cam_id) == 0 || feature->dino_meta.at(cam_id).empty() ||
        feature->timestamps.count(cam_id) == 0 || feature->timestamps.at(cam_id).empty()) {
      continue;
    }

    const std::vector<FeatureDINOMeta> &meta = feature->dino_meta.at(cam_id);
    const std::vector<double> &times = feature->timestamps.at(cam_id);
    TrackState track;
    track.feature_id = feature_id;
    track.class_id = class_id;
    track.inclass_id = feature_id / class_count;
    track.last_bbox = meta.back().bbox;
    track.predicted_center = bbox_center(track.last_bbox);

    // After two stored bboxes we can estimate a constant-velocity center prediction.
    // During bootstrap there is only one displacement sample missing, so velocity is
    // deliberately ignored and the match relies on a looser center gate.
    if (meta.size() >= 2 && times.size() >= 2 && timestamp > times.back() && times.back() > times[times.size() - 2]) {
      const cv::Point2f last_center = bbox_center(meta.back().bbox);
      const cv::Point2f prev_center = bbox_center(meta[meta.size() - 2].bbox);
      const double dt_last = times.back() - times[times.size() - 2];
      const double dt_next = timestamp - times.back();
      const float scale = static_cast<float>(dt_next / dt_last);
      track.predicted_center = last_center + (last_center - prev_center) * scale;
      track.has_velocity = true;
    }
    tracks.push_back(track);
  }

  std::vector<MatchEdge> edges;
  for (size_t det_idx = 0; det_idx < detection_candidates.size(); ++det_idx) {
    const DinoTrackCandidate &det = detection_candidates.at(det_idx);
    const cv::Rect2f &bbox = det.detection.bbox;
    const cv::Point2f center = bbox_center(bbox);
    for (size_t track_idx = 0; track_idx < tracks.size(); ++track_idx) {
      const TrackState &track = tracks.at(track_idx);
      if (track.class_id != det.class_id) {
        continue;
      }

      const float center_dist = point_dist(center, track.predicted_center);
      const float diag_gate = (track.has_velocity ? DINO_MATCH_CENTER_GATE_BBOX_DIAG_MULT : DINO_MATCH_BOOTSTRAP_CENTER_GATE_BBOX_DIAG_MULT) *
                              std::max(rect_diag(bbox), rect_diag(track.last_bbox));
      const float center_gate =
          std::max(track.has_velocity ? DINO_MATCH_MAX_PREDICTED_CENTER_DIST_PX : DINO_MATCH_MAX_BOOTSTRAP_CENTER_DIST_PX, diag_gate);
      if (center_dist > center_gate) {
        continue;
      }

      // Size is useful after the object has a history, but the first two boxes do
      // not yet define a size-change trend. Use a very loose bootstrap gate and a
      // tighter gate once velocity/history exists.
      const float size_cost = size_log_ratio(bbox, track.last_bbox);
      const float size_gate = track.has_velocity ? DINO_MATCH_MAX_SIZE_LOG_RATIO : DINO_MATCH_BOOTSTRAP_MAX_SIZE_LOG_RATIO;
      if (size_cost > size_gate) {
        continue;
      }

      const float iou = bbox_iou(bbox, track.last_bbox);
      const float normalized_center_cost = center_dist / std::max(1.0f, center_gate);
      const float normalized_size_cost = size_cost / std::max(1e-3f, size_gate);
      const float cost = DINO_MATCH_CENTER_COST_WEIGHT * normalized_center_cost + DINO_MATCH_SIZE_COST_WEIGHT * normalized_size_cost +
                         DINO_MATCH_IOU_COST_WEIGHT * (1.0f - iou);
      if (cost <= DINO_MATCH_ACCEPT_COST) {
        edges.push_back({det_idx, track_idx, cost});
      }
    }
  }

  // Build a one-to-one assignment greedily from the best pair costs. This is
  // intentionally simple for v1: DINO detections are sparse, and class gating
  // plus motion/size gates remove most ambiguous pairings before this point.
  std::sort(edges.begin(), edges.end(), [](const MatchEdge &a, const MatchEdge &b) { return a.cost < b.cost; });
  std::vector<bool> detection_matched(detection_candidates.size(), false);
  std::vector<bool> track_matched(tracks.size(), false);
  for (const MatchEdge &edge : edges) {
    if (detection_matched.at(edge.detection_index) || track_matched.at(edge.track_index)) {
      continue;
    }
    DinoTrackCandidate matched = detection_candidates.at(edge.detection_index);
    matched.feature_id = tracks.at(edge.track_index).feature_id;
    matched.inclass_id = tracks.at(edge.track_index).inclass_id;
    candidates.push_back(matched);
    detection_matched.at(edge.detection_index) = true;
    track_matched.at(edge.track_index) = true;
  }

  for (size_t i = 0; i < detection_candidates.size(); ++i) {
    if (detection_matched.at(i)) {
      continue;
    }
    DinoTrackCandidate fresh = detection_candidates.at(i);
    if (!allocate_dino_feature_id(fresh.class_id, class_count, used_feature_ids, used_inclass_ids_by_class, fresh.feature_id, fresh.inclass_id)) {
      PRINT_WARNING("[DINO]: failed to allocate feature id for class '%s'\n", fresh.detection.label.c_str());
      continue;
    }
    candidates.push_back(fresh);
  }

  return candidates;
}

TrackDINO::TrackDINO(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                     HistogramMethod histmethod, TrackDINOConfig config, std::shared_ptr<ov_core::TrackBase> primary_tracker)
    : TrackBase(std::move(cameras), numfeats, numaruco, stereo, histmethod), config_(std::move(config)),
      primary_tracker_(std::move(primary_tracker)) {
  database = std::make_shared<FeatureDatabaseDINO>();

  dino_config backend_config;
  backend_config.engine_path = config_.engine_path;
  backend_config.model = config_.model_path.empty() ? backend_config.model : config_.model_path;
  backend_config.device = config_.device.empty() ? (config_.use_gpu ? "cuda" : "cpu") : config_.device;
  backend_config.verbosity = config_.verbosity;
  backend_.reset(new dino_backend(backend_config, config_.lazy_init));
  PRINT_INFO("[DINO]: tracker created: prompts=%zu box_threshold=%.3f text_threshold=%.3f device=%s lazy_init=%d\n", config_.prompts.size(),
             config_.box_threshold, config_.text_threshold, backend_config.device.c_str(), config_.lazy_init);
}

TrackDINO::~TrackDINO() = default;

void TrackDINO::feed_new_camera(const ov_core::CameraData &message) {
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    throw std::runtime_error("TrackDINO: CameraData sizes do not match");
  }

  for (size_t i = 0; i < message.sensor_ids.size(); ++i) {
    feed_monocular(message, i);
  }
}

std::unordered_map<size_t, std::vector<DinoDetection>> TrackDINO::get_last_detections() const {
  return detections_last_;
}

std::shared_ptr<ov_core::FeatureDatabase> TrackDINO::primary_feature_database() const {
  return primary_tracker_ == nullptr ? nullptr : primary_tracker_->get_feature_database();
}

void TrackDINO::feed_monocular(const ov_core::CameraData &message, size_t msg_id) {
  if (msg_id >= message.images.size() || msg_id >= message.sensor_ids.size() || msg_id >= message.masks.size()) {
    return;
  }

  const size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  const cv::Mat img = preprocess_image(message.images.at(msg_id));
  const std::vector<DinoDetection> detections = backend_->run_dino(img, config_.prompts, config_.box_threshold, config_.text_threshold);
  const std::vector<DinoTrackCandidate> candidates = perform_matching_standalone(detections, cam_id, message.timestamp);
  const std::shared_ptr<FeatureDatabaseDINO> dino_database = std::static_pointer_cast<FeatureDatabaseDINO>(database);

  std::vector<cv::KeyPoint> centers;
  std::vector<size_t> ids;
  centers.reserve(candidates.size());
  ids.reserve(candidates.size());
  for (const DinoTrackCandidate &candidate : candidates) {
    const DinoDetection &detection = candidate.detection;
    const cv::Point2f center(detection.bbox.x + 0.5f * detection.bbox.width, detection.bbox.y + 0.5f * detection.bbox.height);
    const cv::Point2f center_n = camera_calib.at(cam_id)->undistort_cv(center);

    FeatureDINOMeta meta;
    meta.bbox = detection.bbox;
    meta.confidence = detection.confidence;
    meta.label = detection.label;

    dino_database->update_feature_dino(candidate.feature_id, message.timestamp, cam_id, center.x, center.y, center_n.x, center_n.y, meta);
    centers.emplace_back(center, std::max(detection.bbox.width, detection.bbox.height));
    ids.push_back(candidate.feature_id);
  }

  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = message.masks.at(msg_id);
    pts_last[cam_id] = centers;
    ids_last[cam_id] = ids;
    detections_last_[cam_id] = detections;
  }
}

cv::Mat TrackDINO::preprocess_image(const cv::Mat &img) const {
  if (img.empty()) {
    return img;
  }
  if (histogram_method == HistogramMethod::NONE) {
    return img;
  }

  cv::Mat gray;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = img;
  }

  cv::Mat out;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(gray, out);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
    clahe->apply(gray, out);
  } else {
    out = gray;
  }
  return out;
}

} // namespace ov_dino
