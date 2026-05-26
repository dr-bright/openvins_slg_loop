/*
 * Grounding DINO pipe-client image GUI test.
 *
 * Spawns ov_dino/src/dino_engine_server.py, waits for /health, sends one BMP
 * image request to /run_dino, parses JSON detections, and displays boxes.
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string image_path;
  std::string prompt;
  std::string server_path = "/root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py";
  std::string model = "IDEA-Research/grounding-dino-tiny";
  std::string device = "auto";
  std::string save_path;
  float box_threshold = 0.30f;
  float text_threshold = 0.25f;
  int max_detections = 0;
};

struct Detection {
  std::string label;
  float score = 0.0f;
  float x = 0.0f;
  float y = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
};

struct HttpResponse {
  int status = 0;
  std::string reason;
  std::string content_type;
  std::string body;
};

void usage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " --image IMG --prompt TEXT [options]\n"
            << "Options:\n"
            << "  --server PATH          Default: /root/catkin_ws/src/openvins_slg_slam/ov_dino/src/dino_engine_server.py\n"
            << "  --model MODEL          Default: IDEA-Research/grounding-dino-tiny\n"
            << "  --device DEVICE        Default: auto\n"
            << "  --box-threshold X      Default: 0.30\n"
            << "  --text-threshold X     Default: 0.25\n"
            << "  --max-detections N     Default: 0 (unlimited)\n"
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
    } else if (key == "--server") {
      args.server_path = value(key);
    } else if (key == "--model") {
      args.model = value(key);
    } else if (key == "--device") {
      args.device = value(key);
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

std::string url_encode(const std::string &input) {
  std::ostringstream out;
  out << std::uppercase << std::hex;
  for (unsigned char c : input) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
      out << static_cast<char>(c);
    } else {
      out << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(c);
    }
  }
  return out.str();
}

void write_all(int fd, const void *data, size_t size) {
  const char *ptr = static_cast<const char *>(data);
  while (size > 0) {
    const ssize_t written = ::write(fd, ptr, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
    }
    ptr += written;
    size -= static_cast<size_t>(written);
  }
}

std::string read_exact(int fd, size_t size) {
  std::string out(size, '\0');
  size_t offset = 0;
  while (offset < size) {
    const ssize_t n = ::read(fd, &out[offset], size - offset);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("child process closed stdout");
    }
    offset += static_cast<size_t>(n);
  }
  return out;
}

std::string read_until_header_end(int fd) {
  std::string data;
  char c = 0;
  while (data.find("\r\n\r\n") == std::string::npos) {
    const ssize_t n = ::read(fd, &c, 1);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("child process closed stdout while reading headers");
    }
    data.push_back(c);
  }
  return data;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

HttpResponse read_response(int fd) {
  const std::string raw_headers = read_until_header_end(fd);
  std::istringstream ss(raw_headers.substr(0, raw_headers.size() - 4));
  std::string version;
  HttpResponse response;
  ss >> version >> response.status;
  std::getline(ss, response.reason);
  if (!response.reason.empty() && response.reason[0] == ' ') {
    response.reason.erase(response.reason.begin());
  }

  size_t content_length = 0;
  std::string line;
  while (std::getline(ss, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = lower(line.substr(0, colon));
    std::string val = line.substr(colon + 1);
    while (!val.empty() && val.front() == ' ') {
      val.erase(val.begin());
    }
    if (key == "content-length") {
      content_length = static_cast<size_t>(std::stoul(val));
    } else if (key == "content-type") {
      response.content_type = val;
    }
  }
  response.body = read_exact(fd, content_length);
  return response;
}

class DinoChild {
public:
  explicit DinoChild(const Args &args) {
    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
      throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
    }

    pid_ = fork();
    if (pid_ < 0) {
      throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
    }
    if (pid_ == 0) {
      dup2(to_child[0], STDIN_FILENO);
      dup2(from_child[1], STDOUT_FILENO);
      close(to_child[0]);
      close(to_child[1]);
      close(from_child[0]);
      close(from_child[1]);
      execlp("python3", "python3", args.server_path.c_str(), "--model", args.model.c_str(), "--device", args.device.c_str(), "--verbosity",
             "info", nullptr);
      std::cerr << "execlp failed: " << std::strerror(errno) << std::endl;
      _exit(127);
    }

    write_fd_ = to_child[1];
    read_fd_ = from_child[0];
    close(to_child[0]);
    close(from_child[1]);
  }

  ~DinoChild() {
    if (write_fd_ >= 0 && read_fd_ >= 0) {
      try {
        send_request("GET /shutdown HTTP/1.1\r\nContent-Length: 0\r\n\r\n", {});
        (void)read_response(read_fd_);
      } catch (...) {
      }
    }
    if (write_fd_ >= 0) {
      close(write_fd_);
    }
    if (read_fd_ >= 0) {
      close(read_fd_);
    }
    if (pid_ > 0) {
      int status = 0;
      waitpid(pid_, &status, 0);
    }
  }

  void wait_for_health() {
    send_request("GET /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n", {});
    const HttpResponse response = read_response(read_fd_);
    if (response.status != 200) {
      throw std::runtime_error("health check failed: " + std::to_string(response.status) + " " + response.body);
    }
  }

  HttpResponse run_dino(const Args &args, const std::vector<uchar> &bmp) {
    std::ostringstream path;
    path << "/run_dino?box_threshold=" << args.box_threshold << "&text_threshold=" << args.text_threshold << "&max_detections="
         << args.max_detections << "#" << url_encode(args.prompt);

    std::ostringstream header;
    header << "POST " << path.str() << " HTTP/1.1\r\n"
           << "Content-Type: image/bmp\r\n"
           << "Content-Length: " << bmp.size() << "\r\n\r\n";
    send_request(header.str(), bmp);
    return read_response(read_fd_);
  }

private:
  void send_request(const std::string &header, const std::vector<uchar> &body) {
    write_all(write_fd_, header.data(), header.size());
    if (!body.empty()) {
      write_all(write_fd_, body.data(), body.size());
    }
  }

  pid_t pid_ = -1;
  int write_fd_ = -1;
  int read_fd_ = -1;
};

std::string unescape_json_string(const std::string &input) {
  std::string out;
  out.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == '\\' && i + 1 < input.size()) {
      const char c = input[++i];
      if (c == 'n') {
        out.push_back('\n');
      } else if (c == 't') {
        out.push_back('\t');
      } else {
        out.push_back(c);
      }
    } else {
      out.push_back(input[i]);
    }
  }
  return out;
}

std::vector<Detection> parse_detections(const std::string &json) {
  std::vector<Detection> detections;
  const std::regex re(
      R"JSON(\{"image":\d+,"label":"((?:\\.|[^"\\])*)","score":([-+0-9.eE]+),"x":([-+0-9.eE]+),"y":([-+0-9.eE]+),"dx":([-+0-9.eE]+),"dy":([-+0-9.eE]+)\})JSON");
  for (std::sregex_iterator it(json.begin(), json.end(), re), end; it != end; ++it) {
    Detection d;
    d.label = unescape_json_string((*it)[1].str());
    d.score = std::stof((*it)[2].str());
    d.x = std::stof((*it)[3].str());
    d.y = std::stof((*it)[4].str());
    d.dx = std::stof((*it)[5].str());
    d.dy = std::stof((*it)[6].str());
    detections.push_back(d);
  }
  return detections;
}

void draw_detections(cv::Mat &image, const std::vector<Detection> &detections) {
  for (const Detection &d : detections) {
    const int x0 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(d.x))));
    const int y0 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(d.y))));
    const int x1 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(d.x + d.dx))));
    const int y1 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(d.y + d.dy))));
    const cv::Scalar color(0, 220, 255);
    cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1), color, 2);

    std::ostringstream label;
    label << d.label << " " << std::fixed << std::setprecision(2) << d.score;
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

    std::vector<uchar> bmp;
    if (!cv::imencode(".bmp", image, bmp)) {
      throw std::runtime_error("cv::imencode(.bmp) failed");
    }

    DinoChild child(args);
    child.wait_for_health();
    const HttpResponse response = child.run_dino(args, bmp);
    if (response.status != 200) {
      throw std::runtime_error("DINO request failed: " + std::to_string(response.status) + " " + response.body);
    }

    const std::vector<Detection> detections = parse_detections(response.body);
    std::cout << "Detections: " << detections.size() << "\n";
    for (const Detection &d : detections) {
      std::cout << "  " << d.label << " score=" << d.score << " x=" << d.x << " y=" << d.y << " dx=" << d.dx << " dy=" << d.dy << "\n";
    }

    cv::Mat vis = image.clone();
    draw_detections(vis, detections);
    if (!args.save_path.empty()) {
      cv::imwrite(args.save_path, vis);
      std::cout << "Saved: " << args.save_path << "\n";
    }
    cv::imshow("DINO pipe client detections", vis);
    cv::waitKey(0);
  } catch (const std::exception &e) {
    std::cerr << "dino_client_image_gui failed: " << e.what() << "\n";
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
