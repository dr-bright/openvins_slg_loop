/*
 * Manual loop-closure inspection GUI.
 *
 * Usage:
 *   rosrun ov_lightglue manual_loop_closure <bag> <points.pcd> <superpoint.onnx> <lightglue.onnx> [traj.txt] [save_traj.txt] [use_gpu=1]
 *
 * The trajectory arguments are accepted for the future editor workflow and are
 * intentionally unused in this first visual inspection version.
 */

#include "track/slg_backend.h"

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

namespace {

struct VideoFrame {
  double timestamp = 0.0;
  cv::Mat image_bgr;
};

struct OverlayPoint {
  double timestamp = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d p_G = Eigen::Vector3d::Zero();
  double u = 0.0;
  double v = 0.0;
  uint64_t feat_id = 0;
  uint64_t lifetime = 0;
  double score = std::numeric_limits<double>::quiet_NaN();
  cv::Mat descriptor;
};

struct PcdData {
  std::vector<std::string> field_names;
  std::vector<OverlayPoint> points;
  bool has_timestamp = false;
  bool has_u = false;
  bool has_v = false;
  size_t descriptor_dim = 0;
};

struct FramePointSet {
  int frame_index = -1;
  std::vector<size_t> point_indices;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

struct MatchLink {
  size_t point_a = 0;
  size_t point_b = 0;
  uint64_t display_id = 0;
  int sibling_frame_a = -1;
  int sibling_frame_b = -1;
};

struct LoopRequest {
  int frame_a = -1;
  int frame_b = -1;
};

struct TrajectoryPose {
  double timestamp = 0.0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  std::vector<double> tail;
};

struct TrajectoryData {
  std::vector<std::string> header_lines;
  std::vector<TrajectoryPose> poses;
};

struct RenderPatch {
  uint64_t display_id = 0;
  int sibling_frame = -1;
};

template <typename T> T read_as(const std::vector<uint8_t> &data, size_t offset) {
  T value{};
  std::memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

const pcl::PCLPointField *find_field(const pcl::PCLPointCloud2 &cloud, const std::string &name) {
  for (const pcl::PCLPointField &field : cloud.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

double read_numeric_field(const pcl::PCLPointCloud2 &cloud, size_t point_offset, const pcl::PCLPointField &field) {
  const size_t offset = point_offset + field.offset;
  switch (field.datatype) {
  case pcl::PCLPointField::INT8:
    return static_cast<double>(read_as<int8_t>(cloud.data, offset));
  case pcl::PCLPointField::UINT8:
    return static_cast<double>(read_as<uint8_t>(cloud.data, offset));
  case pcl::PCLPointField::INT16:
    return static_cast<double>(read_as<int16_t>(cloud.data, offset));
  case pcl::PCLPointField::UINT16:
    return static_cast<double>(read_as<uint16_t>(cloud.data, offset));
  case pcl::PCLPointField::INT32:
    return static_cast<double>(read_as<int32_t>(cloud.data, offset));
  case pcl::PCLPointField::UINT32:
    return static_cast<double>(read_as<uint32_t>(cloud.data, offset));
  case pcl::PCLPointField::FLOAT32:
    return static_cast<double>(read_as<float>(cloud.data, offset));
  case pcl::PCLPointField::FLOAT64:
    return read_as<double>(cloud.data, offset);
  default:
    return std::numeric_limits<double>::quiet_NaN();
  }
}

uint64_t read_u64_field(const pcl::PCLPointCloud2 &cloud, size_t point_offset, const pcl::PCLPointField *field) {
  if (field == nullptr) {
    return 0;
  }
  const double value = read_numeric_field(cloud, point_offset, *field);
  if (!std::isfinite(value) || value < 0.0) {
    return 0;
  }
  return static_cast<uint64_t>(value);
}

PcdData load_pcd(const std::string &pcd_path) {
  std::cout << "[load] reading PCD: " << pcd_path << std::endl;
  pcl::PCLPointCloud2 cloud;
  if (pcl::io::loadPCDFile(pcd_path, cloud) != 0) {
    throw std::runtime_error("failed to load PCD file: " + pcd_path);
  }

  PcdData pcd;
  pcd.field_names.reserve(cloud.fields.size());
  for (const pcl::PCLPointField &field : cloud.fields) {
    pcd.field_names.push_back(field.name);
  }

  const pcl::PCLPointField *field_u = find_field(cloud, "u");
  const pcl::PCLPointField *field_v = find_field(cloud, "v");
  const pcl::PCLPointField *field_x = find_field(cloud, "x");
  const pcl::PCLPointField *field_y = find_field(cloud, "y");
  const pcl::PCLPointField *field_z = find_field(cloud, "z");
  const pcl::PCLPointField *field_timestamp = find_field(cloud, "timestamp");
  const pcl::PCLPointField *field_feat_id = find_field(cloud, "feat_id");
  if (field_feat_id == nullptr) {
    field_feat_id = find_field(cloud, "featid");
  }
  const pcl::PCLPointField *field_lifetime = find_field(cloud, "lifetime");
  const pcl::PCLPointField *field_score = find_field(cloud, "score");
  std::vector<const pcl::PCLPointField *> descriptor_fields;
  for (const pcl::PCLPointField &field : cloud.fields) {
    if (field.name.find("desc_") == 0) {
      descriptor_fields.push_back(&field);
    }
  }
  std::sort(descriptor_fields.begin(), descriptor_fields.end(), [](const pcl::PCLPointField *a, const pcl::PCLPointField *b) {
    return a->name < b->name;
  });

  pcd.has_u = field_u != nullptr;
  pcd.has_v = field_v != nullptr;
  pcd.has_timestamp = field_timestamp != nullptr;
  pcd.descriptor_dim = descriptor_fields.size();
  if (!pcd.has_timestamp) {
    throw std::runtime_error("PCD has no required timestamp field");
  }
  if (field_x == nullptr || field_y == nullptr || field_z == nullptr) {
    throw std::runtime_error("PCD has no required x/y/z fields");
  }
  if (!pcd.has_u || !pcd.has_v) {
    std::cerr << "PCD has no u/v fields; overlay will be empty until these fields are present." << std::endl;
  }
  if (descriptor_fields.empty()) {
    std::cerr << "PCD has no desc_xxx fields; LightGlue matching will be unavailable." << std::endl;
  }

  const size_t point_count = static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
  std::cout << "[load] decoding PCD points: " << point_count << std::endl;
  pcd.points.reserve(point_count);
  for (size_t i = 0; i < point_count; ++i) {
    const size_t decoded_count = i + 1;
    if (decoded_count % 50000 == 0 || decoded_count == point_count) {
      std::cout << "[load] decoded PCD points " << decoded_count << "/" << point_count << std::endl;
    }
    const size_t point_offset = i * static_cast<size_t>(cloud.point_step);
    if (point_offset + cloud.point_step > cloud.data.size()) {
      break;
    }
    OverlayPoint point;
    point.p_G.x() = read_numeric_field(cloud, point_offset, *field_x);
    point.p_G.y() = read_numeric_field(cloud, point_offset, *field_y);
    point.p_G.z() = read_numeric_field(cloud, point_offset, *field_z);
    if (field_timestamp != nullptr) {
      point.timestamp = read_numeric_field(cloud, point_offset, *field_timestamp);
    }
    if (field_u != nullptr) {
      point.u = read_numeric_field(cloud, point_offset, *field_u);
    }
    if (field_v != nullptr) {
      point.v = read_numeric_field(cloud, point_offset, *field_v);
    }
    if (field_score != nullptr) {
      point.score = read_numeric_field(cloud, point_offset, *field_score);
    }
    if (!descriptor_fields.empty()) {
      point.descriptor.create(1, static_cast<int>(descriptor_fields.size()), CV_32F);
      for (size_t d = 0; d < descriptor_fields.size(); ++d) {
        point.descriptor.at<float>(0, static_cast<int>(d)) =
            static_cast<float>(read_numeric_field(cloud, point_offset, *descriptor_fields.at(d)));
      }
    }
    point.feat_id = read_u64_field(cloud, point_offset, field_feat_id);
    point.lifetime = read_u64_field(cloud, point_offset, field_lifetime);
    pcd.points.push_back(point);
  }

  std::sort(pcd.points.begin(), pcd.points.end(), [](const OverlayPoint &a, const OverlayPoint &b) {
    return a.timestamp < b.timestamp;
  });
  std::cout << "[load] decoded PCD points done: " << pcd.points.size() << std::endl;
  return pcd;
}

cv::Mat decode_bgr(const sensor_msgs::ImageConstPtr &msg) {
  if (!msg) {
    return cv::Mat();
  }
  try {
    if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
      return cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    }
    if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
      cv::Mat rgb = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8)->image;
      cv::Mat bgr;
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      return bgr;
    }
    if (msg->encoding == sensor_msgs::image_encodings::MONO8 || msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1) {
      cv::Mat gray = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8)->image;
      cv::Mat bgr;
      cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
      return bgr;
    }
    if (msg->encoding == sensor_msgs::image_encodings::MONO16 || msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv::Mat mono16 = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16)->image;
      cv::Mat gray;
      mono16.convertTo(gray, CV_8U, 1.0 / 256.0);
      cv::Mat bgr;
      cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
      return bgr;
    }
    return cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  } catch (const std::exception &e) {
    std::cerr << "Image decode failed: " << e.what() << std::endl;
    return cv::Mat();
  }
}

std::string first_image_topic(rosbag::View &view) {
  for (const rosbag::MessageInstance &m : view) {
    sensor_msgs::ImageConstPtr image = m.instantiate<sensor_msgs::Image>();
    if (image != nullptr) {
      return m.getTopic();
    }
  }
  return "";
}

std::vector<VideoFrame> load_video_frames(const std::string &bag_path) {
  std::cout << "[load] reading bag: " << bag_path << std::endl;
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View all_view(bag);

  const std::string image_topic = first_image_topic(all_view);
  if (image_topic.empty()) {
    throw std::runtime_error("bag has no sensor_msgs/Image messages");
  }
  std::cout << "Using image topic: " << image_topic << std::endl;

  rosbag::View image_view(bag, rosbag::TopicQuery(image_topic));
  std::vector<VideoFrame> frames;
  const size_t total_messages = image_view.size();
  std::cout << "[load] decoding image frames: " << total_messages << std::endl;
  for (const rosbag::MessageInstance &m : image_view) {
    sensor_msgs::ImageConstPtr image_msg = m.instantiate<sensor_msgs::Image>();
    if (image_msg == nullptr) {
      continue;
    }
    cv::Mat image = decode_bgr(image_msg);
    if (image.empty()) {
      continue;
    }
    VideoFrame frame;
    frame.timestamp = image_msg->header.stamp.toSec();
    frame.image_bgr = image;
    frames.push_back(std::move(frame));
    if (frames.size() % 100 == 0 || frames.size() == total_messages) {
      std::cout << "[load] decoded image frames " << frames.size() << "/" << total_messages << std::endl;
    }
  }
  std::cout << "[load] decoded image frames done: " << frames.size() << "/" << total_messages << std::endl;
  bag.close();
  if (frames.empty()) {
    throw std::runtime_error("no decodable image frames in bag");
  }
  return frames;
}

double infer_fps(const std::vector<VideoFrame> &frames) {
  std::vector<double> dts;
  for (size_t i = 1; i < frames.size() && i < 300; ++i) {
    const double dt = frames.at(i).timestamp - frames.at(i - 1).timestamp;
    if (dt > 1e-6) {
      dts.push_back(dt);
    }
  }
  if (dts.empty()) {
    return 30.0;
  }
  std::sort(dts.begin(), dts.end());
  return 1.0 / dts.at(dts.size() / 2);
}

std::pair<size_t, size_t> timestamp_range(const PcdData &pcd, double timestamp, double tolerance_sec) {
  const double t0 = timestamp - tolerance_sec;
  const double t1 = timestamp + tolerance_sec;
  auto lower = std::lower_bound(pcd.points.begin(), pcd.points.end(), t0, [](const OverlayPoint &point, double time) {
    return point.timestamp < time;
  });
  auto upper = std::upper_bound(pcd.points.begin(), pcd.points.end(), t1, [](double time, const OverlayPoint &point) {
    return time < point.timestamp;
  });
  return {static_cast<size_t>(std::distance(pcd.points.begin(), lower)), static_cast<size_t>(std::distance(pcd.points.begin(), upper))};
}

FramePointSet collect_frame_points(const PcdData &pcd, int frame_index, double timestamp, double tolerance_sec) {
  FramePointSet set;
  set.frame_index = frame_index;
  const auto range = timestamp_range(pcd, timestamp, tolerance_sec);
  for (size_t i = range.first; i < range.second; ++i) {
    const OverlayPoint &point = pcd.points.at(i);
    if (!std::isfinite(point.u) || !std::isfinite(point.v) || point.descriptor.empty()) {
      continue;
    }
    set.point_indices.push_back(i);
    cv::KeyPoint keypoint;
    keypoint.pt = cv::Point2f(static_cast<float>(point.u), static_cast<float>(point.v));
    keypoint.size = 1.0f;
    keypoint.response = std::isfinite(point.score) ? static_cast<float>(point.score) : 0.0f;
    set.keypoints.push_back(keypoint);
    set.descriptors.push_back(point.descriptor);
  }
  return set;
}

uint64_t link_display_id(const OverlayPoint &a, const OverlayPoint &b) {
  if (a.feat_id != 0) {
    return a.feat_id;
  }
  return b.feat_id;
}

std::vector<MatchLink> run_frame_matching(const ov_lightglue::slg_backend &backend, const PcdData &pcd, const std::vector<VideoFrame> &frames,
                                          int frame_a, int frame_b, double tolerance_sec, float min_confidence) {
  const FramePointSet a = collect_frame_points(pcd, frame_a, frames.at(frame_a).timestamp, tolerance_sec);
  const FramePointSet b = collect_frame_points(pcd, frame_b, frames.at(frame_b).timestamp, tolerance_sec);
  if (a.keypoints.empty() || b.keypoints.empty() || a.descriptors.empty() || b.descriptors.empty()) {
    return {};
  }

  std::vector<cv::DMatch> matches;
  backend.run_lightglue(frames.at(frame_a).image_bgr.size(), a.keypoints, a.descriptors, frames.at(frame_b).image_bgr.size(), b.keypoints,
                        b.descriptors, matches, min_confidence, false);

  std::vector<MatchLink> links;
  links.reserve(matches.size());
  for (const cv::DMatch &match : matches) {
    if (match.queryIdx < 0 || match.trainIdx < 0 || match.queryIdx >= static_cast<int>(a.point_indices.size()) ||
        match.trainIdx >= static_cast<int>(b.point_indices.size())) {
      continue;
    }
    MatchLink link;
    link.point_a = a.point_indices.at(static_cast<size_t>(match.queryIdx));
    link.point_b = b.point_indices.at(static_cast<size_t>(match.trainIdx));
    link.display_id = link_display_id(pcd.points.at(link.point_a), pcd.points.at(link.point_b));
    link.sibling_frame_a = frame_b;
    link.sibling_frame_b = frame_a;
    links.push_back(link);
  }
  return links;
}

uint64_t max_feature_id(const PcdData &pcd) {
  uint64_t max_id = 0;
  for (const OverlayPoint &point : pcd.points) {
    max_id = std::max(max_id, point.feat_id);
  }
  return max_id;
}

void replace_feature_id(PcdData &pcd, uint64_t old_id, uint64_t new_id) {
  if (old_id == 0 || new_id == 0 || old_id == new_id) {
    return;
  }
  for (OverlayPoint &point : pcd.points) {
    if (point.feat_id == old_id) {
      point.feat_id = new_id;
    }
  }
}

void replace_display_id(std::vector<MatchLink> &links, uint64_t old_id, uint64_t new_id) {
  if (old_id == 0 || new_id == 0 || old_id == new_id) {
    return;
  }
  for (MatchLink &link : links) {
    if (link.display_id == old_id) {
      link.display_id = new_id;
    }
  }
}

void confirm_pending_matches(PcdData &pcd, std::vector<MatchLink> &confirmed_matches, const std::vector<MatchLink> &pending_matches,
                             std::vector<LoopRequest> &loop_requests, const LoopRequest &request, uint64_t &next_synthetic_id) {
  if (pending_matches.empty()) {
    return;
  }

  loop_requests.push_back(request);

  for (MatchLink link : pending_matches) {
    OverlayPoint &a = pcd.points.at(link.point_a);
    OverlayPoint &b = pcd.points.at(link.point_b);
    uint64_t kept_id = link.display_id;
    if (kept_id == 0) {
      kept_id = next_synthetic_id++;
    }

    const uint64_t old_a = a.feat_id;
    const uint64_t old_b = b.feat_id;
    if (old_a != 0 && old_a != kept_id) {
      replace_feature_id(pcd, old_a, kept_id);
      replace_display_id(confirmed_matches, old_a, kept_id);
    }
    if (old_b != 0 && old_b != kept_id) {
      replace_feature_id(pcd, old_b, kept_id);
      replace_display_id(confirmed_matches, old_b, kept_id);
    }
    a.feat_id = kept_id;
    b.feat_id = kept_id;
    link.display_id = kept_id;
    confirmed_matches.push_back(link);
  }
}

std::unordered_map<size_t, RenderPatch> build_render_patches(const PcdData &pcd, const std::vector<MatchLink> &confirmed,
                                                             const std::vector<MatchLink> &pending) {
  std::unordered_map<size_t, RenderPatch> patches;
  std::unordered_map<uint64_t, int> confirmed_sibling_by_id;
  auto add_link = [&patches](const MatchLink &link) {
    RenderPatch patch_a;
    patch_a.display_id = link.display_id;
    patch_a.sibling_frame = link.sibling_frame_a;
    patches[link.point_a] = patch_a;
    RenderPatch patch_b;
    patch_b.display_id = link.display_id;
    patch_b.sibling_frame = link.sibling_frame_b;
    patches[link.point_b] = patch_b;
  };
  for (const MatchLink &link : confirmed) {
    add_link(link);
    if (link.display_id != 0) {
      confirmed_sibling_by_id[link.display_id] = link.sibling_frame_a;
    }
  }
  for (size_t i = 0; i < pcd.points.size(); ++i) {
    const uint64_t feat_id = pcd.points.at(i).feat_id;
    auto sibling_it = confirmed_sibling_by_id.find(feat_id);
    if (sibling_it == confirmed_sibling_by_id.end()) {
      continue;
    }
    RenderPatch patch;
    patch.display_id = feat_id;
    patch.sibling_frame = sibling_it->second;
    patches[i] = patch;
  }
  for (const MatchLink &link : pending) {
    add_link(link);
  }
  return patches;
}

size_t draw_overlay(cv::Mat &image, const PcdData &pcd, double timestamp, double tolerance_sec,
                    const std::unordered_map<size_t, RenderPatch> &render_patches) {
  if (!pcd.has_u || !pcd.has_v) {
    return 0;
  }
  const auto range = timestamp_range(pcd, timestamp, tolerance_sec);
  size_t drawn = 0;
  for (size_t i = range.first; i < range.second; ++i) {
    const OverlayPoint &point = pcd.points.at(i);
    if (!std::isfinite(point.u) || !std::isfinite(point.v)) {
      continue;
    }
    const int u = static_cast<int>(std::round(point.u));
    const int v = static_cast<int>(std::round(point.v));
    if (u < 0 || v < 0 || u >= image.cols || v >= image.rows) {
      continue;
    }
    const auto patch_it = render_patches.find(i);
    const bool patched = patch_it != render_patches.end();
    cv::Scalar color = patched ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 255);
    if (!patched && point.lifetime > 0) {
      const int green = std::min<int>(255, 80 + static_cast<int>(point.lifetime) * 12);
      color = cv::Scalar(0, green, 255 - green / 2);
    }
    cv::circle(image, cv::Point(u, v), 3, color, cv::FILLED, cv::LINE_AA);
    const uint64_t label_id = patched ? patch_it->second.display_id : point.feat_id;
    if (label_id != 0) {
      std::string label = std::to_string(label_id);
      if (patched) {
        label += " f" + std::to_string(patch_it->second.sibling_frame + 1);
      }
      cv::putText(image, label, cv::Point(u + 4, v - 4), cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv::LINE_AA);
    }
    ++drawn;
  }
  return drawn;
}

std::string key_name_hint() {
  return "space pause | f/r play | .,/ step | z/x jump10 | [] speed | h help | g goto";
}

std::vector<std::string> extended_key_hints() {
  return {
      "Shift+1..9 bind bookmark | 1..9 seek bookmark | g opens goto-frame prompt",
      "m then 1..9 match current frame with bookmark | Enter confirm | \\ abort pending",
      "z/x jump -/+10 frames, also < > if backend reports them | w wipe rollback, also | if backend reports it",
      "s save corrected trajectory | p preview current correction using temp trajectory | g goto frame | q/Esc quit",
  };
}

int shifted_digit_slot(int key) {
  const std::string shifted = "!@#$%^&*(";
  const size_t pos = shifted.find(static_cast<char>(key));
  return pos == std::string::npos ? -1 : static_cast<int>(pos);
}

int digit_slot(int key) {
  if (key >= '1' && key <= '9') {
    return key - '1';
  }
  return -1;
}

int ctrl_digit_slot(int key) {
  // OpenCV highgui does not expose modifiers portably. These codes work in some
  // terminals/backends; unsupported backends will simply never enter this path.
  if (key >= 0x11 && key <= 0x19) {
    return key - 0x11;
  }
  return -1;
}

TrajectoryData load_trajectory(const std::string &path) {
  TrajectoryData trajectory;
  if (path.empty()) {
    return trajectory;
  }
  std::ifstream in(path.c_str());
  if (!in.is_open()) {
    throw std::runtime_error("failed to open trajectory: " + path);
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    if (line.at(0) == '#') {
      trajectory.header_lines.push_back(line);
      continue;
    }
    std::istringstream ss(line);
    TrajectoryPose pose;
    double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
    if (!(ss >> pose.timestamp >> pose.p.x() >> pose.p.y() >> pose.p.z() >> qx >> qy >> qz >> qw)) {
      continue;
    }
    pose.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    double value = 0.0;
    while (ss >> value) {
      pose.tail.push_back(value);
    }
    trajectory.poses.push_back(std::move(pose));
  }
  return trajectory;
}

void save_trajectory(const std::string &path, const TrajectoryData &trajectory) {
  if (path.empty()) {
    throw std::runtime_error("save trajectory path is empty");
  }
  std::ofstream out(path.c_str());
  if (!out.is_open()) {
    throw std::runtime_error("failed to open save trajectory path: " + path);
  }

  for (const std::string &header : trajectory.header_lines) {
    out << header << "\n";
  }
  for (const TrajectoryPose &pose : trajectory.poses) {
    out << std::fixed << std::setprecision(5) << pose.timestamp;
    out << std::setprecision(6) << " " << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " " << pose.q.x() << " " << pose.q.y()
        << " " << pose.q.z() << " " << pose.q.w();
    out << std::setprecision(10);
    for (double value : pose.tail) {
      out << " " << value;
    }
    out << "\n";
  }
}

std::vector<size_t> frame_point_indices_with_ids(const PcdData &pcd, const std::vector<VideoFrame> &frames, int frame_index, double tolerance_sec) {
  const auto range = timestamp_range(pcd, frames.at(frame_index).timestamp, tolerance_sec);
  std::vector<size_t> indices;
  for (size_t i = range.first; i < range.second; ++i) {
    if (pcd.points.at(i).feat_id != 0 && pcd.points.at(i).p_G.allFinite()) {
      indices.push_back(i);
    }
  }
  return indices;
}

bool estimate_loop_transform(const PcdData &pcd, const std::vector<VideoFrame> &frames, const LoopRequest &request, double tolerance_sec,
                             Eigen::Matrix4d &T_target_source, size_t &pair_count) {
  const int source_frame = std::max(request.frame_a, request.frame_b);
  const int target_frame = std::min(request.frame_a, request.frame_b);
  const std::vector<size_t> source_indices = frame_point_indices_with_ids(pcd, frames, source_frame, tolerance_sec);
  const std::vector<size_t> target_indices = frame_point_indices_with_ids(pcd, frames, target_frame, tolerance_sec);

  std::unordered_map<uint64_t, size_t> target_by_id;
  for (size_t index : target_indices) {
    target_by_id[pcd.points.at(index).feat_id] = index;
  }

  pcl::PointCloud<pcl::PointXYZ> source_cloud;
  pcl::PointCloud<pcl::PointXYZ> target_cloud;
  for (size_t source_index : source_indices) {
    const uint64_t feat_id = pcd.points.at(source_index).feat_id;
    auto target_it = target_by_id.find(feat_id);
    if (target_it == target_by_id.end()) {
      continue;
    }
    const Eigen::Vector3d &source = pcd.points.at(source_index).p_G;
    const Eigen::Vector3d &target = pcd.points.at(target_it->second).p_G;
    source_cloud.push_back(pcl::PointXYZ(static_cast<float>(source.x()), static_cast<float>(source.y()), static_cast<float>(source.z())));
    target_cloud.push_back(pcl::PointXYZ(static_cast<float>(target.x()), static_cast<float>(target.y()), static_cast<float>(target.z())));
  }

  pair_count = source_cloud.size();
  if (pair_count < 3) {
    return false;
  }

  Eigen::Matrix4f T_float = Eigen::Matrix4f::Identity();
  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float> estimator;
  estimator.estimateRigidTransformation(source_cloud, target_cloud, T_float);
  T_target_source = T_float.cast<double>();
  return T_target_source.allFinite();
}

size_t first_trajectory_index_at_or_after(const TrajectoryData &trajectory, double timestamp) {
  auto it = std::lower_bound(trajectory.poses.begin(), trajectory.poses.end(), timestamp, [](const TrajectoryPose &pose, double time) {
    return pose.timestamp < time;
  });
  return static_cast<size_t>(std::distance(trajectory.poses.begin(), it));
}

void apply_transform_to_trajectory_tail(TrajectoryData &trajectory, size_t start_index, const Eigen::Matrix4d &T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  const Eigen::Quaterniond q_R(R);
  for (size_t i = start_index; i < trajectory.poses.size(); ++i) {
    trajectory.poses.at(i).p = R * trajectory.poses.at(i).p + t;
    trajectory.poses.at(i).q = (q_R * trajectory.poses.at(i).q).normalized();
  }
}

void perform_correction_and_save(const PcdData &pcd, const std::vector<VideoFrame> &frames, const std::vector<LoopRequest> &loop_requests,
                                 const std::string &traj_path, const std::string &save_traj_path, double tolerance_sec) {
  if (traj_path.empty() || save_traj_path.empty()) {
    std::cout << "Save ignored: traj.txt and save_traj.txt must be provided" << std::endl;
    return;
  }
  TrajectoryData trajectory = load_trajectory(traj_path);
  if (trajectory.poses.empty()) {
    std::cout << "Save ignored: trajectory is empty" << std::endl;
    return;
  }
  if (loop_requests.empty()) {
    save_trajectory(save_traj_path, trajectory);
    std::cout << "Saved unchanged trajectory: " << save_traj_path << std::endl;
    return;
  }

  std::vector<LoopRequest> sorted_requests = loop_requests;
  std::sort(sorted_requests.begin(), sorted_requests.end(), [](const LoopRequest &a, const LoopRequest &b) {
    return std::max(a.frame_a, a.frame_b) < std::max(b.frame_a, b.frame_b);
  });

  size_t applied = 0;
  for (const LoopRequest &request : sorted_requests) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    size_t pair_count = 0;
    if (!estimate_loop_transform(pcd, frames, request, tolerance_sec, T, pair_count)) {
      std::cout << "Loop request frames " << (request.frame_a + 1) << " <-> " << (request.frame_b + 1)
                << " skipped: insufficient/invalid 3D pairs (" << pair_count << ")" << std::endl;
      continue;
    }
    const int source_frame = std::max(request.frame_a, request.frame_b);
    const double source_time = frames.at(source_frame).timestamp;
    const size_t start_index = first_trajectory_index_at_or_after(trajectory, source_time);
    if (start_index >= trajectory.poses.size()) {
      std::cout << "Loop request frames " << (request.frame_a + 1) << " <-> " << (request.frame_b + 1)
                << " skipped: no trajectory sample after source frame" << std::endl;
      continue;
    }
    apply_transform_to_trajectory_tail(trajectory, start_index, T);
    ++applied;
    const Eigen::Vector3d trans = T.block<3, 1>(0, 3);
    std::cout << "Applied loop request frames " << (request.frame_a + 1) << " <-> " << (request.frame_b + 1) << " using " << pair_count
              << " pairs, translation_norm=" << trans.norm() << std::endl;
  }

  save_trajectory(save_traj_path, trajectory);
  std::cout << "Saved corrected trajectory to " << save_traj_path << " using " << applied << "/" << sorted_requests.size()
            << " loop requests" << std::endl;
}

std::string shell_quote(const std::string &text) {
  std::string quoted = "'";
  for (char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

void preview_trajectory(const PcdData &pcd, const std::vector<VideoFrame> &frames, const std::vector<LoopRequest> &loop_requests,
                        const std::string &traj_path, double tolerance_sec) {
  if (traj_path.empty()) {
    std::cout << "Preview ignored: traj.txt must be provided" << std::endl;
    return;
  }

  char tmp_template[] = "/tmp/manual_loop_closure_preview_XXXXXX.txt";
  const int fd = mkstemps(tmp_template, 4);
  if (fd < 0) {
    std::cout << "Preview ignored: failed to create temporary trajectory file" << std::endl;
    return;
  }
  close(fd);
  const std::string temp_path = tmp_template;

  perform_correction_and_save(pcd, frames, loop_requests, traj_path, temp_path, tolerance_sec);
  const std::string command = "rosrun ov_eval plot_trajectories posyaw " + shell_quote(traj_path) + " " + shell_quote(temp_path);
  std::cout << "Preview: " << command << std::endl;
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cout << "Preview command returned " << rc << std::endl;
  }
  if (std::remove(temp_path.c_str()) != 0) {
    std::cout << "Preview warning: failed to remove temporary trajectory file: " << temp_path << std::endl;
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 8) {
    std::cerr << "Usage: " << argv[0] << " <bag> <points.pcd> <superpoint.onnx> <lightglue.onnx> [traj.txt] [save_traj.txt] [use_gpu=1]"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string bag_path = argv[1];
  const std::string pcd_path = argv[2];
  const std::string superpoint_path = argv[3];
  const std::string lightglue_path = argv[4];
  const std::string traj_path = argc > 5 ? argv[5] : "";
  const std::string save_traj_path = argc > 6 ? argv[6] : "";
  const bool use_gpu = argc > 7 ? std::atoi(argv[7]) != 0 : true;
  (void)traj_path;
  (void)save_traj_path;

  try {
    std::vector<VideoFrame> frames = load_video_frames(bag_path);
    PcdData pcd = load_pcd(pcd_path);
    const PcdData original_pcd = pcd;
    ov_lightglue::slg_backend backend(superpoint_path, lightglue_path, use_gpu, ov_lightglue::slg_backend::log_level::warning);
    const double fps = infer_fps(frames);
    const double tolerance_sec = std::max(0.5 / fps, 1e-3);

    std::cout << "Loaded frames: " << frames.size() << " fps~" << fps << std::endl;
    std::cout << "Loaded PCD points: " << pcd.points.size() << std::endl;
    std::cout << "PCD fields:";
    for (const std::string &name : pcd.field_names) {
      std::cout << " " << name;
    }
    std::cout << std::endl;

    const std::string window_name = "manual_loop_closure";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    int frame_index = 0;
    int direction = 1;
    bool paused = true;
    double speed = 1.0;
    std::map<int, int> bookmarks;
    std::vector<MatchLink> pending_matches;
    std::vector<MatchLink> confirmed_matches;
    std::vector<LoopRequest> loop_requests;
    LoopRequest pending_loop_request;
    bool has_pending_loop_request = false;
    uint64_t next_synthetic_id = max_feature_id(pcd) + 1;
    bool awaiting_match_slot = false;
    bool show_extended_help = false;
    bool goto_mode = false;
    std::string goto_buffer;

    while (true) {
      frame_index = std::max(0, std::min(frame_index, static_cast<int>(frames.size()) - 1));
      cv::Mat display = frames.at(frame_index).image_bgr.clone();
      const std::unordered_map<size_t, RenderPatch> render_patches = build_render_patches(pcd, confirmed_matches, pending_matches);
      const size_t overlay_count = draw_overlay(display, pcd, frames.at(frame_index).timestamp, tolerance_sec, render_patches);

      std::ostringstream status;
      status << "frame " << frame_index + 1 << "/" << frames.size() << " t=" << std::fixed << std::setprecision(3)
             << frames.at(frame_index).timestamp << " overlay=" << overlay_count << " " << (paused ? "paused" : "playing")
             << " dir=" << direction << " speed=" << std::setprecision(2) << speed << " pending=" << pending_matches.size()
             << " confirmed=" << confirmed_matches.size();
      cv::putText(display, status.str(), cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
      cv::putText(display, status.str(), cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::putText(display, key_name_hint(), cv::Point(12, display.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3,
                  cv::LINE_AA);
      cv::putText(display, key_name_hint(), cv::Point(12, display.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      if (show_extended_help) {
        const std::vector<std::string> hints = extended_key_hints();
        int y = display.rows - 12 - static_cast<int>(hints.size()) * 20;
        for (const std::string &hint : hints) {
          cv::putText(display, hint, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
          cv::putText(display, hint, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          y += 20;
        }
      }
      if (goto_mode) {
        const std::string prompt = "Goto frame: " + goto_buffer + "_";
        const int y = std::max(54, display.rows - 92);
        cv::rectangle(display, cv::Point(8, y - 24), cv::Point(std::min(display.cols - 8, 360), y + 8), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(display, prompt, cv::Point(14, y), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      }

      cv::imshow(window_name, display);

      const int delay_ms = paused ? 0 : std::max(1, static_cast<int>(1000.0 / (fps * speed)));
      const int key = cv::waitKeyEx(delay_ms);

      if (goto_mode) {
        if (key >= '0' && key <= '9') {
          goto_buffer.push_back(static_cast<char>(key));
        } else if (key == '\\') {
          if (!goto_buffer.empty()) {
            goto_buffer.pop_back();
          }
        } else if (key == 10 || key == 13) {
          if (!goto_buffer.empty()) {
            const int requested_frame = std::atoi(goto_buffer.c_str());
            if (requested_frame > 0) {
              frame_index = requested_frame - 1;
            }
          }
          goto_mode = false;
          goto_buffer.clear();
        } else if (key == 27) {
          goto_mode = false;
          goto_buffer.clear();
        } else if (key == -1 && !paused) {
          frame_index += direction;
        }
      } else if (key == 'q' || key == 27) {
        break;
      } else if (key == ' ') {
        paused = !paused;
      } else if (key == 'f') {
        direction = 1;
        paused = false;
      } else if (key == 'r') {
        direction = -1;
        paused = false;
      } else if (key == 'p') {
        preview_trajectory(pcd, frames, loop_requests, traj_path, tolerance_sec);
      } else if (key == 'h') {
        show_extended_help = !show_extended_help;
      } else if (key == 'g') {
        goto_mode = true;
        goto_buffer.clear();
        show_extended_help = false;
        paused = true;
      } else if (key == '.' || key == 83 || key == 2555904) {
        paused = true;
        frame_index++;
      } else if (key == ',' || key == 81 || key == 2424832) {
        paused = true;
        frame_index--;
      } else if (key == '>' || key == 'x') {
        paused = true;
        frame_index += 10;
      } else if (key == '<' || key == 'z') {
        paused = true;
        frame_index -= 10;
      } else if (key == '[') {
        speed = std::max(0.125, speed * 0.5);
      } else if (key == ']') {
        speed = std::min(16.0, speed * 2.0);
      } else if (key == 'm') {
        awaiting_match_slot = true;
        std::cout << "Match mode: press 1-9 to match current frame against a bookmarked frame" << std::endl;
      } else if (shifted_digit_slot(key) >= 0) {
        const int slot = shifted_digit_slot(key);
        bookmarks[slot] = frame_index;
        std::cout << "Bound slot " << (slot + 1) << " to frame " << frame_index << " t=" << frames.at(frame_index).timestamp << std::endl;
      } else if (ctrl_digit_slot(key) >= 0) {
        const int slot = ctrl_digit_slot(key);
        pending_matches.clear();
        has_pending_loop_request = false;
        auto it = bookmarks.find(slot);
        if (it == bookmarks.end()) {
          std::cout << "Ctrl+" << (slot + 1) << " ignored: slot is not bound" << std::endl;
        } else if (it->second == frame_index) {
          std::cout << "Ctrl+" << (slot + 1) << " ignored: bookmarked frame is current frame" << std::endl;
        } else {
          pending_matches = run_frame_matching(backend, pcd, frames, frame_index, it->second, tolerance_sec, 0.5f);
          pending_loop_request = LoopRequest{frame_index, it->second};
          has_pending_loop_request = true;
          paused = true;
          std::cout << "Pending match current frame " << (frame_index + 1) << " <-> slot " << (slot + 1) << " frame "
                    << (it->second + 1) << ": " << pending_matches.size() << " matches" << std::endl;
        }
      } else if (digit_slot(key) >= 0) {
        const int slot = digit_slot(key);
        if (awaiting_match_slot) {
          awaiting_match_slot = false;
          pending_matches.clear();
          has_pending_loop_request = false;
          auto it = bookmarks.find(slot);
          if (it == bookmarks.end()) {
            std::cout << "Match ignored: slot " << (slot + 1) << " is not bound" << std::endl;
          } else if (it->second == frame_index) {
            std::cout << "Match ignored: bookmarked frame is current frame" << std::endl;
          } else {
            pending_matches = run_frame_matching(backend, pcd, frames, frame_index, it->second, tolerance_sec, 0.5f);
            pending_loop_request = LoopRequest{frame_index, it->second};
            has_pending_loop_request = true;
            paused = true;
            std::cout << "Pending match current frame " << (frame_index + 1) << " <-> slot " << (slot + 1) << " frame "
                      << (it->second + 1) << ": " << pending_matches.size() << " matches" << std::endl;
          }
        } else {
          auto it = bookmarks.find(slot);
          if (it != bookmarks.end()) {
            frame_index = it->second;
            paused = true;
          }
        }
      } else if (key == '\\') {
        if (!pending_matches.empty()) {
          std::cout << "Aborted " << pending_matches.size() << " pending matches" << std::endl;
        }
        pending_matches.clear();
        has_pending_loop_request = false;
      } else if (key == '|' || key == 'w') {
        pcd = original_pcd;
        pending_matches.clear();
        confirmed_matches.clear();
        loop_requests.clear();
        has_pending_loop_request = false;
        awaiting_match_slot = false;
        next_synthetic_id = max_feature_id(pcd) + 1;
        std::cout << "Rolled back all confirmed matches and loop requests" << std::endl;
      } else if (key == 10 || key == 13) {
        if (!pending_matches.empty() && has_pending_loop_request) {
          const size_t count = pending_matches.size();
          confirm_pending_matches(pcd, confirmed_matches, pending_matches, loop_requests, pending_loop_request, next_synthetic_id);
          std::cout << "Confirmed " << count << " matches; total confirmed=" << confirmed_matches.size()
                    << ", loop requests=" << loop_requests.size() << std::endl;
          pending_matches.clear();
          has_pending_loop_request = false;
        }
      } else if (key == 's') {
        perform_correction_and_save(pcd, frames, loop_requests, traj_path, save_traj_path, tolerance_sec);
      } else if (key != -1) {
        if (awaiting_match_slot) {
          awaiting_match_slot = false;
          std::cout << "Match mode cancelled" << std::endl;
        }
        if (!paused) {
          frame_index += direction;
        }
      } else if (!paused) {
        frame_index += direction;
      }

      if (frame_index < 0) {
        frame_index = 0;
        paused = true;
      }
      if (frame_index >= static_cast<int>(frames.size())) {
        frame_index = static_cast<int>(frames.size()) - 1;
        paused = true;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "manual_loop_closure failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
