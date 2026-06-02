/*
 * Grounding DINO backend image GUI test.
 */

#include "track/dino_backend.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string image_path;
  std::string prompt;
  std::string engine_path = "/root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py";
  std::string model = "IDEA-Research/grounding-dino-tiny";
  std::string device = "auto";
  std::string verbosity = "info";
  std::string save_path;
  float box_threshold = 0.30f;
  float text_threshold = 0.25f;
  int max_detections = 0;
};

void usage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " --image IMG --prompt TEXT [options]\n"
            << "Options:\n"
            << "  --engine PATH          Default: /root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py\n"
            << "  --model MODEL          Default: IDEA-Research/grounding-dino-tiny\n"
            << "  --device DEVICE        Default: auto\n"
            << "  --verbosity LEVEL      Default: info\n"
            << "  --box-threshold X      Default: 0.30\n"
            << "  --text-threshold X     Default: 0.25\n"
            << "  --max-detections N     Default: 0\n"
            << "  --save PATH            Optional annotated image output\n";
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (key == "--image" || key == "-i") {
      args.image_path = value(key);
    } else if (key == "--prompt" || key == "-p") {
      args.prompt = value(key);
    } else if (key == "--engine") {
      args.engine_path = value(key);
    } else if (key == "--model") {
      args.model = value(key);
    } else if (key == "--device") {
      args.device = value(key);
    } else if (key == "--verbosity") {
      args.verbosity = value(key);
    } else if (key == "--box-threshold") {
      args.box_threshold = std::stof(value(key));
    } else if (key == "--text-threshold") {
      args.text_threshold = std::stof(value(key));
    } else if (key == "--max-detections") {
      args.max_detections = std::stoi(value(key));
    } else if (key == "--save") {
      args.save_path = value(key);
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.image_path.empty()) {
    std::cout << "Image path: ";
    std::getline(std::cin, args.image_path);
  }
  if (args.prompt.empty()) {
    std::cout << "Text prompt: ";
    std::getline(std::cin, args.prompt);
  }
  return args;
}

std::vector<std::string> split_classes(const std::string &prompt) {
  std::vector<std::string> classes;
  std::string current;
  for (char c : prompt) {
    if (c == '.') {
      if (!current.empty()) {
        classes.push_back(current);
      }
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    classes.push_back(current);
  }
  return classes;
}

void draw_detections(cv::Mat &image, const std::vector<ov_dino::dino_detection> &detections) {
  for (const ov_dino::dino_detection &d : detections) {
    const int x0 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(d.bbox.x))));
    const int y0 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(d.bbox.y))));
    const int x1 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(d.bbox.x + d.bbox.width))));
    const int y1 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(d.bbox.y + d.bbox.height))));
    const cv::Scalar color(0, 220, 255);
    cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1), color, 2);

    std::ostringstream label;
    label << d.label << " " << std::fixed << std::setprecision(2) << d.confidence;
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
    const int ly0 = std::max(0, y0 - ts.height - baseline - 4);
    cv::rectangle(image, cv::Point(x0, ly0), cv::Point(std::min(image.cols - 1, x0 + ts.width + 6), y0), color, -1);
    cv::putText(image, label.str(), cv::Point(x0 + 3, y0 - baseline - 2), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    cv::Mat image = cv::imread(args.image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to read image: " + args.image_path);
    }

    ov_dino::dino_config config;
    config.engine_path = args.engine_path;
    config.model = args.model;
    config.device = args.device;
    config.verbosity = args.verbosity;
    ov_dino::dino_backend backend(config, false);

    const std::vector<ov_dino::dino_detection> detections =
        backend.run_dino(image, split_classes(args.prompt), args.box_threshold, args.text_threshold, args.max_detections);

    std::cout << "Detections: " << detections.size() << "\n";
    for (const auto &d : detections) {
      std::cout << "  " << d.label << " score=" << d.confidence << " x=" << d.bbox.x << " y=" << d.bbox.y << " dx=" << d.bbox.width
                << " dy=" << d.bbox.height << "\n";
    }

    cv::Mat vis = image.clone();
    draw_detections(vis, detections);
    if (!args.save_path.empty()) {
      cv::imwrite(args.save_path, vis);
      std::cout << "Saved: " << args.save_path << "\n";
    }
    cv::imshow("DINO backend detections", vis);
    cv::waitKey(0);
  } catch (const std::exception &e) {
    std::cerr << "dino_backend_image_gui failed: " << e.what() << "\n";
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
