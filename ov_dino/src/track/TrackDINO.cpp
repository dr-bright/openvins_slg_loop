/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#include "track/TrackDINO.h"

#include "cam/CamBase.h"
#include "track/FeatureDatabaseDINO.h"
#include "utils/print.h"

#include <algorithm>
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

struct DinoTrackCandidate {
  DinoDetection detection;
  size_t class_id = 0;
  size_t inclass_id = 0;
  size_t feature_id = 0;
};

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

std::vector<DinoTrackCandidate> assign_detection_ids_stub(const std::vector<DinoDetection> &detections, const std::vector<std::string> &classes,
                                                          const std::shared_ptr<ov_core::FeatureDatabase> &database) {
  std::vector<DinoTrackCandidate> candidates;
  if (classes.empty()) {
    return candidates;
  }

  const size_t class_count = classes.size();
  const std::map<std::string, size_t> class_lookup = build_class_lookup(classes);
  std::set<size_t> used_feature_ids;
  std::map<size_t, std::set<size_t>> used_inclass_ids_by_class;
  collect_existing_ids(database, class_count, used_feature_ids, used_inclass_ids_by_class);

  candidates.reserve(detections.size());
  for (const DinoDetection &detection : detections) {
    const auto class_it = class_lookup.find(normalize_label(detection.label));
    if (class_it == class_lookup.end()) {
      PRINT_DEBUG("[DINO]: skipping detection with unknown class label '%s'\n", detection.label.c_str());
      continue;
    }

    DinoTrackCandidate candidate;
    candidate.detection = detection;
    candidate.class_id = class_it->second;
    if (!allocate_dino_feature_id(candidate.class_id, class_count, used_feature_ids, used_inclass_ids_by_class, candidate.feature_id,
                                  candidate.inclass_id)) {
      PRINT_WARNING("[DINO]: failed to allocate feature id for class '%s'\n", detection.label.c_str());
      continue;
    }
    candidates.push_back(candidate);
  }

  return candidates;
}

} // namespace

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
  const std::vector<DinoTrackCandidate> candidates = assign_detection_ids_stub(detections, config_.prompts, database);
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
