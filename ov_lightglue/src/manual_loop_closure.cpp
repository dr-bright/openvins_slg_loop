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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>

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

struct RenderPatch {
  uint64_t display_id = 0;
  int sibling_frame = -1;
};

struct StdinSeekState {
  std::atomic<bool> done{false};
  std::atomic<int> requested_frame{-1};
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
  if (!pcd.has_u || !pcd.has_v) {
    std::cerr << "PCD has no u/v fields; overlay will be empty until these fields are present." << std::endl;
  }
  if (descriptor_fields.empty()) {
    std::cerr << "PCD has no desc_xxx fields; LightGlue matching will be unavailable." << std::endl;
  }

  const size_t point_count = static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
  pcd.points.reserve(point_count);
  for (size_t i = 0; i < point_count; ++i) {
    const size_t point_offset = i * static_cast<size_t>(cloud.point_step);
    if (point_offset + cloud.point_step > cloud.data.size()) {
      break;
    }
    OverlayPoint point;
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
  }
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

std::unordered_map<size_t, RenderPatch> build_render_patches(const std::vector<MatchLink> &confirmed, const std::vector<MatchLink> &pending) {
  std::unordered_map<size_t, RenderPatch> patches;
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
  return "space pause | p/r play | .,/ step | [] speed | Shift+1..9 bind | 1..9 seek | m then 1..9 match | Enter confirm | Backspace abort";
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

void stdin_seek_thread(StdinSeekState *state) {
  while (!state->done.load()) {
    std::cout << "frame> " << std::flush;
    int frame = -1;
    if (!(std::cin >> frame)) {
      state->done.store(true);
      break;
    }
    state->requested_frame.store(frame);
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
    const PcdData pcd = load_pcd(pcd_path);
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
    bool awaiting_match_slot = false;
    std::shared_ptr<StdinSeekState> stdin_state = std::make_shared<StdinSeekState>();
    std::thread input_thread([stdin_state]() {
      stdin_seek_thread(stdin_state.get());
    });
    input_thread.detach();

    while (true) {
      const int requested_frame = stdin_state->requested_frame.exchange(-1);
      if (requested_frame > 0) {
        frame_index = requested_frame - 1;
        paused = true;
      }

      frame_index = std::max(0, std::min(frame_index, static_cast<int>(frames.size()) - 1));
      cv::Mat display = frames.at(frame_index).image_bgr.clone();
      const std::unordered_map<size_t, RenderPatch> render_patches = build_render_patches(confirmed_matches, pending_matches);
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

      cv::imshow(window_name, display);

      const int delay_ms = paused ? 0 : std::max(1, static_cast<int>(1000.0 / (fps * speed)));
      const int key = cv::waitKeyEx(delay_ms);

      if (key == 'q' || key == 27) {
        break;
      } else if (key == ' ') {
        paused = !paused;
      } else if (key == 'p') {
        direction = 1;
        paused = false;
      } else if (key == 'r') {
        direction = -1;
        paused = false;
      } else if (key == '.' || key == 83 || key == 2555904) {
        paused = true;
        frame_index++;
      } else if (key == ',' || key == 81 || key == 2424832) {
        paused = true;
        frame_index--;
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
        auto it = bookmarks.find(slot);
        if (it == bookmarks.end()) {
          std::cout << "Ctrl+" << (slot + 1) << " ignored: slot is not bound" << std::endl;
        } else if (it->second == frame_index) {
          std::cout << "Ctrl+" << (slot + 1) << " ignored: bookmarked frame is current frame" << std::endl;
        } else {
          pending_matches = run_frame_matching(backend, pcd, frames, frame_index, it->second, tolerance_sec, 0.5f);
          paused = true;
          std::cout << "Pending match current frame " << (frame_index + 1) << " <-> slot " << (slot + 1) << " frame "
                    << (it->second + 1) << ": " << pending_matches.size() << " matches" << std::endl;
        }
      } else if (digit_slot(key) >= 0) {
        const int slot = digit_slot(key);
        if (awaiting_match_slot) {
          awaiting_match_slot = false;
          pending_matches.clear();
          auto it = bookmarks.find(slot);
          if (it == bookmarks.end()) {
            std::cout << "Match ignored: slot " << (slot + 1) << " is not bound" << std::endl;
          } else if (it->second == frame_index) {
            std::cout << "Match ignored: bookmarked frame is current frame" << std::endl;
          } else {
            pending_matches = run_frame_matching(backend, pcd, frames, frame_index, it->second, tolerance_sec, 0.5f);
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
      } else if (key == 8 || key == 127) {
        if (!pending_matches.empty()) {
          std::cout << "Aborted " << pending_matches.size() << " pending matches" << std::endl;
        }
        pending_matches.clear();
      } else if (key == 10 || key == 13) {
        if (!pending_matches.empty()) {
          confirmed_matches.insert(confirmed_matches.end(), pending_matches.begin(), pending_matches.end());
          std::cout << "Confirmed " << pending_matches.size() << " matches; total confirmed=" << confirmed_matches.size() << std::endl;
          pending_matches.clear();
        }
      } else if (key == 'S') {
        std::cout << "Shift+S save/correction placeholder: procedure not defined yet" << std::endl;
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
    stdin_state->done.store(true);
  } catch (const std::exception &e) {
    std::cerr << "manual_loop_closure failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
