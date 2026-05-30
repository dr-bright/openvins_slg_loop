/*
 * openvins_lightglue: SLG feature metadata.
 */

#include "track/FeatureSLG.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ov_lightglue {

void FeatureSLG::export_latest_metadata(std::map<std::string, uint64_t> &u64_fields, std::map<std::string, int64_t> &i64_fields,
                                        std::map<std::string, double> &f64_fields) const {
  (void)u64_fields;
  (void)i64_fields;

  struct LatestObservation {
    bool valid = false;
    size_t cam_id = 0;
    size_t index = 0;
    double timestamp = -std::numeric_limits<double>::infinity();
  };

  LatestObservation latest;
  for (const auto &cam_descriptors : descriptors) {
    const size_t cam_id = cam_descriptors.first;
    const std::vector<cv::Mat> &desc_vec = cam_descriptors.second;
    const auto timestamps_it = timestamps.find(cam_id);
    if (timestamps_it == timestamps.end()) {
      continue;
    }

    const size_t count = std::min(desc_vec.size(), timestamps_it->second.size());
    for (size_t i = 0; i < count; ++i) {
      if (desc_vec.at(i).empty()) {
        continue;
      }
      const double timestamp = timestamps_it->second.at(i);
      if (!latest.valid || timestamp > latest.timestamp) {
        latest.valid = true;
        latest.cam_id = cam_id;
        latest.index = i;
        latest.timestamp = timestamp;
      }
    }
  }

  if (!latest.valid) {
    return;
  }

  const cv::Mat &descriptor = descriptors.at(latest.cam_id).at(latest.index);
  const auto scores_it = scores.find(latest.cam_id);
  if (scores_it != scores.end() && latest.index < scores_it->second.size()) {
    f64_fields["score"] = static_cast<double>(scores_it->second.at(latest.index));
  }

  cv::Mat descriptor_contiguous = descriptor.isContinuous() ? descriptor : descriptor.clone();
  cv::Mat descriptor_f64;
  descriptor_contiguous.reshape(1, 1).convertTo(descriptor_f64, CV_64F);
  for (int i = 0; i < descriptor_f64.cols; ++i) {
    std::ostringstream name;
    name << "desc_" << std::setw(3) << std::setfill('0') << i;
    f64_fields[name.str()] = descriptor_f64.at<double>(0, i);
  }
}

} // namespace ov_lightglue
