/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#ifndef OV_DINO_TRACK_DINO_CONFIG_H
#define OV_DINO_TRACK_DINO_CONFIG_H

#include <string>
#include <vector>

namespace ov_dino {

struct TrackDINOConfig {
  TrackDINOConfig();

  std::string model_path;
  std::string engine_path = "/root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py";
  std::string device = "auto";
  std::string verbosity = "warning";
  std::vector<std::string> prompts;
  float box_threshold = 0.30f;
  float text_threshold = 0.25f;
  bool use_gpu = true;
  bool lazy_init = true;
};

} // namespace ov_dino

#endif // OV_DINO_TRACK_DINO_CONFIG_H
