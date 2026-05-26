/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#ifndef OV_DINO_TRACK_DINO_H
#define OV_DINO_TRACK_DINO_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "track/TrackBase.h"
#include "track/TrackDINOConfig.h"
#include "track/dino_backend.h"

namespace ov_dino {

class TrackDINO : public ov_core::TrackBase {
public:
  explicit TrackDINO(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                     HistogramMethod histmethod, TrackDINOConfig config, std::shared_ptr<ov_core::TrackBase> primary_tracker = nullptr);

  ~TrackDINO() override;

  void feed_new_camera(const ov_core::CameraData &message) override;

  std::unordered_map<size_t, std::vector<DinoDetection>> get_last_detections() const;

protected:
  struct DinoTrackCandidate {
    DinoDetection detection;
    size_t class_id = 0;
    size_t inclass_id = 0;
    size_t feature_id = 0;
  };

  void feed_monocular(const ov_core::CameraData &message, size_t msg_id);
  cv::Mat preprocess_image(const cv::Mat &img) const;
  std::shared_ptr<ov_core::FeatureDatabase> primary_feature_database() const;
  std::vector<DinoTrackCandidate> perform_matching_standalone(const std::vector<DinoDetection> &detections, size_t cam_id,
                                                              double timestamp) const;

protected:
  std::unique_ptr<dino_backend> backend_;
  TrackDINOConfig config_;
  std::shared_ptr<ov_core::TrackBase> primary_tracker_;
  std::unordered_map<size_t, std::vector<DinoDetection>> detections_last_;
};

} // namespace ov_dino

#endif // OV_DINO_TRACK_DINO_H
