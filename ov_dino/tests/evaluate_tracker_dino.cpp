/*
 * Rosbag to MP4 visualization tool.
 *
  rosrun ov_dino evaluate_tracker_dino \
    --bag /data/gopro10/slow_fast/1440.bag \
    --save /data/gopro10/slow_fast/openvins_slg_slam/dino_eval.mp4 \
    --csv /data/gopro10/slow_fast/openvins_slg_slam/dino_metrics.csv \
    --prompt "bright triangle. ceiling support pillar" \
    --fps 5
 */

#include "cam/CamRadtan.h"
#include "feat/FeatureDatabase.h"
#include "track/FeatureDINO.h"
#include "track/TrackDINO.h"
#include "utils/sensor_data.h"

#include <Eigen/Core>

#include <algorithm>
#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct Args {
  std::string model = "IDEA-Research/grounding-dino-tiny";
  std::string engine = "/root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py";
  bool use_gpu = true;
  std::string bag;
  std::string topic = "/cam0/image_raw";
  std::string save;
  std::string csv;
  size_t max_generation = 16;
  std::string prompt = "bright triangle";
  float box_threshold = 0.25f;
  float text_threshold = 0.25f;
  double start = 0.0;
  double stop = 0.0;
  double fps = 0.0;
};

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " --bag BAG --save OUTPUT.mp4 [options]\n"
            << "\nOptions:\n"
            << "  --bag PATH             Required rosbag input.\n"
            << "  --save PATH            Required output MP4 path.\n"
            << "  --csv PATH             Optional metrics CSV path.\n"
            << "  --topic TOPIC          Image topic. Default: /cam0/image_raw\n"
            << "  --model NAME_OR_PATH   HF model. Default: IDEA-Research/grounding-dino-tiny\n"
            << "  --engine PATH          Python engine. Default: ov_dino/src/dino_engine_server.py\n"
            << "  --gpu 0|1              Use CUDA engine device. Default: 1\n"
            << "  --prompt TEXT          Dot-separated DINO prompt/classes.\n"
            << "  --box FLOAT            Box threshold. Default: 0.30\n"
            << "  --text FLOAT           Text threshold. Default: 0.25\n"
            << "  --P INT                Max generation bucket. Default: 16\n"
            << "  --start SEC            Start time offset from first selected message. Default: 0\n"
            << "  --stop SEC             Stop time offset from first selected message. Default: end\n"
            << "  --fps FLOAT            Inference/output FPS. Default: infer from bag\n";
}

bool parse_bool_int(const std::string &value) {
  return std::atoi(value.c_str()) != 0;
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (key.empty() || key[0] != '-') {
      throw std::runtime_error("unexpected positional argument: " + key);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + key);
    }
    const std::string value = argv[++i];
    if (key == "--bag") {
      args.bag = value;
    } else if (key == "--save") {
      args.save = value;
    } else if (key == "--csv") {
      args.csv = value;
    } else if (key == "--topic") {
      args.topic = value;
    } else if (key == "--model") {
      args.model = value;
    } else if (key == "--engine") {
      args.engine = value;
    } else if (key == "--gpu" || key == "--use-gpu") {
      args.use_gpu = parse_bool_int(value);
    } else if (key == "--prompt") {
      args.prompt = value;
    } else if (key == "--box" || key == "--box-threshold") {
      args.box_threshold = std::atof(value.c_str());
    } else if (key == "--text" || key == "--text-threshold") {
      args.text_threshold = std::atof(value.c_str());
    } else if (key == "--P" || key == "--max-generation") {
      const int parsed = std::atoi(value.c_str());
      if (parsed > 0) {
        args.max_generation = static_cast<size_t>(parsed);
      }
    } else if (key == "--start") {
      args.start = std::atof(value.c_str());
    } else if (key == "--stop") {
      args.stop = std::atof(value.c_str());
    } else if (key == "--fps") {
      args.fps = std::atof(value.c_str());
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (args.bag.empty()) {
    throw std::runtime_error("--bag is required");
  }
  if (args.save.empty()) {
    throw std::runtime_error("--save is required");
  }
  if (args.stop > 0.0 && args.stop <= args.start) {
    throw std::runtime_error("--stop must be greater than --start");
  }
  if (args.fps < 0.0) {
    throw std::runtime_error("--fps must be non-negative");
  }
  return args;
}

cv::Mat decode_grayscale(const sensor_msgs::ImageConstPtr &msg) {
  if (!msg) {
    return cv::Mat();
  }

  cv::Mat gray;
  try {
    if (msg->encoding == sensor_msgs::image_encodings::MONO8 || msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1) {
      gray = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8)->image;
    } else if (msg->encoding == sensor_msgs::image_encodings::MONO16 || msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv::Mat mono16 = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16)->image;
      mono16.convertTo(gray, CV_8U, 1.0 / 256.0);
    } else {
      cv::Mat bgr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
      cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    }
  } catch (...) {
    return cv::Mat();
  }

  return gray;
}

std::string trim_copy(const std::string &text) {
  size_t first = 0;
  while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) {
    ++first;
  }
  size_t last = text.size();
  while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) {
    --last;
  }
  return text.substr(first, last - first);
}

std::vector<std::string> split_classes(const std::string &prompt) {
  std::vector<std::string> classes;
  std::stringstream ss(prompt);
  std::string item;
  while (std::getline(ss, item, '.')) {
    item = trim_copy(item);
    if (!item.empty()) {
      classes.push_back(item);
    }
  }
  if (classes.empty()) {
    classes.push_back(prompt);
  }
  return classes;
}

void draw_overlay(cv::Mat &frame_bgr, const std::vector<size_t> &curr_ids, const std::vector<cv::KeyPoint> &curr_kpts,
                  const std::unordered_map<size_t, cv::Point2f> &prev_pts_by_id, size_t frame_index) {
  size_t carried = 0;

  for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
    const size_t id = curr_ids[i];
    const cv::Point2f curr = curr_kpts[i].pt;

    auto it = prev_pts_by_id.find(id);
    if (it != prev_pts_by_id.end()) {
      ++carried;
      cv::line(frame_bgr, it->second, curr, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
      cv::circle(frame_bgr, curr, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
    } else {
      cv::circle(frame_bgr, curr, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }
  }

  const size_t active = curr_ids.size();
  const size_t fresh = (active >= carried) ? (active - carried) : 0;
  const std::string text = "frame=" + std::to_string(frame_index) + " active=" + std::to_string(active) + " carried=" +
                           std::to_string(carried) + " new=" + std::to_string(fresh);
  cv::putText(frame_bgr, text, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

void draw_dino_boxes(cv::Mat &frame_bgr, const std::shared_ptr<ov_core::FeatureDatabase> &database, const std::vector<size_t> &curr_ids,
                     size_t cam_id) {
  if (database == nullptr) {
    return;
  }

  for (size_t id : curr_ids) {
    std::shared_ptr<ov_core::Feature> base = database->get_feature(id, false);
    std::shared_ptr<ov_dino::FeatureDINO> feat = std::dynamic_pointer_cast<ov_dino::FeatureDINO>(base);
    if (feat == nullptr || feat->dino_meta.count(cam_id) == 0 || feat->dino_meta.at(cam_id).empty()) {
      continue;
    }

    const ov_dino::FeatureDINOMeta &meta = feat->dino_meta.at(cam_id).back();
    const int x0 = std::max(0, std::min(frame_bgr.cols - 1, static_cast<int>(std::round(meta.bbox.x))));
    const int y0 = std::max(0, std::min(frame_bgr.rows - 1, static_cast<int>(std::round(meta.bbox.y))));
    const int x1 = std::max(0, std::min(frame_bgr.cols - 1, static_cast<int>(std::round(meta.bbox.x + meta.bbox.width))));
    const int y1 = std::max(0, std::min(frame_bgr.rows - 1, static_cast<int>(std::round(meta.bbox.y + meta.bbox.height))));
    if (x1 <= x0 || y1 <= y0) {
      continue;
    }

    cv::rectangle(frame_bgr, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    std::ostringstream label;
    label.setf(std::ios::fixed);
    label.precision(2);
    label << meta.label << " " << meta.confidence;
    const int text_y = std::max(16, y0 - 5);
    cv::putText(frame_bgr, label.str(), cv::Point(x0, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(frame_bgr, label.str(), cv::Point(x0, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1,
                cv::LINE_AA);
  }
}

double infer_fps_from_bag(rosbag::View &view) {
  std::vector<double> timestamps;
  timestamps.reserve(300);
  for (const rosbag::MessageInstance &m : view) {
    sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
    if (!img_msg) {
      continue;
    }
    timestamps.push_back(img_msg->header.stamp.toSec());
    if (timestamps.size() >= 300) {
      break;
    }
  }

  std::vector<double> dts;
  dts.reserve(timestamps.size());
  for (size_t i = 1; i < timestamps.size(); ++i) {
    const double dt = timestamps[i] - timestamps[i - 1];
    if (dt > 1e-6) {
      dts.push_back(dt);
    }
  }

  if (dts.empty()) {
    return 0.0;
  }

  std::sort(dts.begin(), dts.end());
  const double median_dt = dts[dts.size() / 2];
  if (median_dt <= 1e-6) {
    return 0.0;
  }
  return std::round(1.0 / median_dt);
}

ros::Time first_image_time(rosbag::View &view) {
  for (const rosbag::MessageInstance &m : view) {
    sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
    if (img_msg) {
      return img_msg->header.stamp;
    }
  }
  return ros::Time();
}

std::unique_ptr<ov_core::TrackBase> create_tracker(const cv::Mat &img_gray, const std::string &model_path, const std::string &engine_path,
                                                   bool use_gpu, const std::vector<std::string> &classes, float box_threshold,
                                                   float text_threshold) {
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
  auto cam = std::make_shared<ov_core::CamRadtan>(img_gray.cols, img_gray.rows);
  Eigen::Matrix<double, 8, 1> calib;
  calib << 300.0, 300.0, static_cast<double>(img_gray.cols) * 0.5, static_cast<double>(img_gray.rows) * 0.5, 0.0, 0.0, 0.0, 0.0;
  cam->set_value(calib);
  cameras.insert({0, cam});

  ov_dino::TrackDINOConfig cfg;
  cfg.model_path = model_path;
  cfg.engine_path = engine_path;
  cfg.use_gpu = use_gpu;
  cfg.device = use_gpu ? "cuda" : "cpu";
  cfg.verbosity = "info";
  cfg.prompts = classes;
  cfg.box_threshold = box_threshold;
  cfg.text_threshold = text_threshold;
  cfg.lazy_init = false;

  return std::unique_ptr<ov_core::TrackBase>(
      new ov_dino::TrackDINO(cameras, 300, 0, false, ov_core::TrackBase::HistogramMethod::NONE, cfg));
}

} // namespace

int main(int argc, char **argv) {
  Args args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Argument error: " << e.what() << "\n";
    print_usage(argv[0]);
    return 2;
  }
  const std::vector<std::string> classes = split_classes(args.prompt);

  try {
    std::cout << "DINO evaluator config:\n"
              << "  model=" << args.model << "\n"
              << "  engine=" << args.engine << "\n"
              << "  use_gpu=" << args.use_gpu << "\n"
              << "  bag=" << args.bag << "\n"
              << "  topic=" << args.topic << "\n"
              << "  save=" << args.save << "\n"
              << "  csv=" << args.csv << "\n"
              << "  P=" << args.max_generation << "\n"
              << "  prompt=" << args.prompt << "\n"
              << "  classes=" << classes.size() << "\n"
              << "  box_threshold=" << args.box_threshold << "\n"
              << "  text_threshold=" << args.text_threshold << "\n"
              << "  start=" << args.start << "\n"
              << "  stop=" << args.stop << "\n"
              << "  fps=" << args.fps << std::endl;
    for (size_t i = 0; i < classes.size(); ++i) {
      std::cout << "    class[" << i << "]=" << classes.at(i) << std::endl;
    }

    rosbag::Bag bag;
    bag.open(args.bag, rosbag::bagmode::Read);
    rosbag::View full_view(bag, rosbag::TopicQuery(args.topic));

    if (full_view.size() == 0) {
      std::cerr << "No messages found for topic: " << args.topic << std::endl;
      return 3;
    }

    const ros::Time first_time = first_image_time(full_view);
    if (first_time.isZero()) {
      std::cerr << "No decodable sensor_msgs/Image frames on topic: " << args.topic << std::endl;
      return 3;
    }

    const ros::Time start_time = first_time + ros::Duration(args.start);
    const ros::Time stop_time = args.stop > 0.0 ? first_time + ros::Duration(args.stop) : full_view.getEndTime();
    rosbag::View view(bag, rosbag::TopicQuery(args.topic), start_time, stop_time);
    if (view.size() == 0) {
      std::cerr << "No messages found in requested slice for topic: " << args.topic << std::endl;
      return 3;
    }

    double out_fps = args.fps;
    if (out_fps <= 0.0) {
      out_fps = infer_fps_from_bag(view);
    }
    if (out_fps <= 0.0) {
      std::cerr << "Unable to infer FPS from rosbag timestamps." << std::endl;
      return 3;
    }
    std::cout << "Using output/inference fps=" << out_fps << std::endl;

    bool tracker_initialized = false;
    bool writer_initialized = false;

    std::unique_ptr<ov_core::TrackBase> tracker;
    std::unordered_map<size_t, cv::Point2f> prev_pts_by_id;
    std::deque<std::unordered_set<size_t>> id_history;
    cv::VideoWriter writer;
    std::ofstream metrics;
    if (!args.csv.empty()) {
      metrics.open(args.csv);
      if (!metrics.is_open()) {
        std::cerr << "Failed to open metrics csv: " << args.csv << std::endl;
        return 4;
      }
      metrics << "frame,timestamp";
      for (size_t g = 0; g <= args.max_generation; ++g) {
        metrics << ",gen" << g;
      }
      metrics << "\n";
    }

    size_t frame_idx = 0;
    size_t used_msgs = 0;
    double next_process_time = -std::numeric_limits<double>::infinity();
    const double process_dt = out_fps > 0.0 ? 1.0 / out_fps : 0.0;

    for (const rosbag::MessageInstance &m : view) {
      sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
      if (!img_msg) {
        continue;
      }
      const double stamp = img_msg->header.stamp.toSec();
      if (stamp + 1e-9 < next_process_time) {
        continue;
      }
      next_process_time = stamp + process_dt;

      cv::Mat img_gray = decode_grayscale(img_msg);
      if (img_gray.empty()) {
        continue;
      }

      if (!tracker_initialized) {
        tracker = create_tracker(img_gray, args.model, args.engine, args.use_gpu, classes, args.box_threshold, args.text_threshold);
        tracker_initialized = true;
      }

      if (!writer_initialized) {
        writer.open(args.save, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), out_fps, img_gray.size(), true);
        if (!writer.isOpened()) {
          std::cerr << "Failed to open output writer: " << args.save << std::endl;
          return 4;
        }
        writer_initialized = true;
      }

      ov_core::CameraData data;
      data.timestamp = stamp;
      data.sensor_ids.push_back(0);
      data.images.push_back(img_gray);
      data.masks.push_back(cv::Mat::zeros(img_gray.rows, img_gray.cols, CV_8UC1));
      tracker->feed_new_camera(data);

      const auto ids_by_cam = tracker->get_last_ids();
      const auto obs_by_cam = tracker->get_last_obs();
      const std::vector<size_t> curr_ids = (ids_by_cam.count(0) > 0) ? ids_by_cam.at(0) : std::vector<size_t>{};
      const std::vector<cv::KeyPoint> curr_kpts = (obs_by_cam.count(0) > 0) ? obs_by_cam.at(0) : std::vector<cv::KeyPoint>{};

      std::unordered_set<size_t> curr_id_set;
      curr_id_set.reserve(curr_ids.size());
      for (size_t id : curr_ids) {
        curr_id_set.insert(id);
      }

      // at_least[g] = number of current features that are continuously present for >= g frames in the past.
      // Build this by chained intersections to prevent discontinuity overcounting if IDs are reused.
      std::vector<size_t> at_least(args.max_generation + 1, 0);
      std::unordered_set<size_t> alive_chain = curr_id_set;
      for (size_t g = 1; g <= args.max_generation; ++g) {
        if (id_history.size() < g) {
          break;
        }
        const auto &ids_g_back = id_history[id_history.size() - g];
        std::unordered_set<size_t> next_chain;
        next_chain.reserve(alive_chain.size());
        for (size_t id : alive_chain) {
          if (ids_g_back.find(id) != ids_g_back.end()) {
            next_chain.insert(id);
          }
        }
        alive_chain.swap(next_chain);
        at_least[g] = alive_chain.size();
      }

      // gen0 = newly appeared this frame.
      // genN (1..P-1) = exact lifetime N (non-overlapping bins).
      // genP = tail bucket for lifetime >= P.
      std::vector<size_t> gen_counts(args.max_generation + 1, 0);
      gen_counts[0] = curr_id_set.size() - at_least[1];
      for (size_t g = 1; g < args.max_generation; ++g) {
        gen_counts[g] = at_least[g] - at_least[g + 1];
      }
      gen_counts[args.max_generation] = at_least[args.max_generation];

      if (metrics.is_open()) {
        metrics << frame_idx << "," << data.timestamp;
        for (size_t g = 0; g <= args.max_generation; ++g) {
          metrics << "," << gen_counts[g];
        }
        metrics << "\n";
      }

      cv::Mat frame_bgr;
      cv::cvtColor(img_gray, frame_bgr, cv::COLOR_GRAY2BGR);
      draw_overlay(frame_bgr, curr_ids, curr_kpts, prev_pts_by_id, frame_idx);
      draw_dino_boxes(frame_bgr, tracker->get_feature_database(), curr_ids, 0);
      writer.write(frame_bgr);

      prev_pts_by_id.clear();
      prev_pts_by_id.reserve(curr_ids.size());
      for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
        prev_pts_by_id[curr_ids[i]] = curr_kpts[i].pt;
      }

      id_history.push_back(std::move(curr_id_set));
      if (id_history.size() > args.max_generation) {
        id_history.pop_front();
      }

      ++frame_idx;
      ++used_msgs;
      if (frame_idx % 50 == 0) {
        std::cout << "processed_frames=" << frame_idx << std::endl;
      }
    }

    if (used_msgs == 0) {
      std::cerr << "No decodable sensor_msgs/Image frames on topic: " << args.topic << std::endl;
      return 3;
    }

    writer.release();
    if (metrics.is_open()) {
      metrics.close();
    }
    bag.close();

    std::cout << "Done. frames_written=" << frame_idx << " output=" << args.save;
    if (!args.csv.empty()) {
      std::cout << " metrics=" << args.csv;
    }
    std::cout << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "evaluate_tracker_dino failed: " << e.what() << std::endl;
    return 5;
  }

  return 0;
}
