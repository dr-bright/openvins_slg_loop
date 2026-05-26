/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#ifndef OV_DINO_TRACK_DINO_BACKEND_H
#define OV_DINO_TRACK_DINO_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace ov_dino {

struct dino_config {
  std::string engine_path = "/root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py";
  std::string model = "IDEA-Research/grounding-dino-tiny";
  std::string device = "auto";
  std::string verbosity = "warning";
};

struct dino_detection {
  std::string label;
  float confidence = 0.0f;
  cv::Rect2f bbox;
};

using DinoDetection = dino_detection;

class dino_backend {
public:
  explicit dino_backend(dino_config config, bool lazy_init = true);
  ~dino_backend();

  dino_backend(const dino_backend &) = delete;
  dino_backend &operator=(const dino_backend &) = delete;
  dino_backend(dino_backend &&) noexcept;
  dino_backend &operator=(dino_backend &&) noexcept;

  std::vector<dino_detection> run_dino(const cv::Mat &image, const std::vector<std::string> &classes, float box_threshold = 0.30f,
                                       float text_threshold = 0.25f, int max_detections = 0);

  void await_ready();
  bool is_ready() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

} // namespace ov_dino

#endif // OV_DINO_TRACK_DINO_BACKEND_H
