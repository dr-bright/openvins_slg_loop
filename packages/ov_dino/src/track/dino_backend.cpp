/*
 * ov_dino: text-prompted object detection extension for OpenVINS
 */

#include "track/dino_backend.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace ov_dino {

namespace {

struct http_response {
  int status = 0;
  std::string reason;
  std::string body;
};

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

std::string prompt_from_classes(const std::vector<std::string> &classes) {
  std::ostringstream prompt;
  bool first = true;
  for (std::string label : classes) {
    label.erase(label.begin(), std::find_if(label.begin(), label.end(), [](unsigned char c) { return !std::isspace(c); }));
    label.erase(std::find_if(label.rbegin(), label.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), label.end());
    if (label.empty()) {
      continue;
    }
    if (!first) {
      prompt << ' ';
    }
    prompt << label;
    if (label.back() != '.') {
      prompt << '.';
    }
    first = false;
  }
  return prompt.str();
}

void write_all(int fd, const void *data, size_t size) {
  const char *ptr = static_cast<const char *>(data);
  while (size > 0) {
    const ssize_t written = ::write(fd, ptr, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("dino_backend: write failed: ") + std::strerror(errno));
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
      throw std::runtime_error(std::string("dino_backend: read failed: ") + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("dino_backend: engine closed stdout");
    }
    offset += static_cast<size_t>(n);
  }
  return out;
}

std::string read_headers(int fd) {
  std::string data;
  char c = 0;
  while (data.find("\r\n\r\n") == std::string::npos) {
    const ssize_t n = ::read(fd, &c, 1);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("dino_backend: read failed: ") + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("dino_backend: engine closed stdout while reading response headers");
    }
    data.push_back(c);
  }
  return data;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

http_response read_response(int fd) {
  const std::string headers = read_headers(fd);
  std::istringstream ss(headers.substr(0, headers.size() - 4));
  std::string version;
  http_response response;
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
    std::string value = line.substr(colon + 1);
    while (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }
    if (key == "content-length") {
      content_length = static_cast<size_t>(std::stoul(value));
    }
  }
  response.body = read_exact(fd, content_length);
  return response;
}

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

std::vector<dino_detection> parse_detections(const std::string &json) {
  std::vector<dino_detection> detections;
  const std::regex re(
      R"JSON(\{"image":\d+,"label":"((?:\\.|[^"\\])*)","score":([-+0-9.eE]+),"x":([-+0-9.eE]+),"y":([-+0-9.eE]+),"dx":([-+0-9.eE]+),"dy":([-+0-9.eE]+)\})JSON");
  for (std::sregex_iterator it(json.begin(), json.end(), re), end; it != end; ++it) {
    dino_detection det;
    det.label = unescape_json_string((*it)[1].str());
    det.confidence = std::stof((*it)[2].str());
    det.bbox.x = std::stof((*it)[3].str());
    det.bbox.y = std::stof((*it)[4].str());
    det.bbox.width = std::stof((*it)[5].str());
    det.bbox.height = std::stof((*it)[6].str());
    detections.push_back(det);
  }
  return detections;
}

} // namespace

struct dino_backend::impl {
  impl(dino_config cfg, bool lazy) : config(std::move(cfg)), lazy_init(lazy) {
    start_engine();
    if (!lazy_init) {
      await_ready();
    }
  }

  ~impl() {
    shutdown();
  }

  void start_engine() {
    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
      throw std::runtime_error(std::string("dino_backend: pipe failed: ") + std::strerror(errno));
    }

    pid = fork();
    if (pid < 0) {
      throw std::runtime_error(std::string("dino_backend: fork failed: ") + std::strerror(errno));
    }
    if (pid == 0) {
      dup2(to_child[0], STDIN_FILENO);
      dup2(from_child[1], STDOUT_FILENO);
      close(to_child[0]);
      close(to_child[1]);
      close(from_child[0]);
      close(from_child[1]);
      execlp("python3", "python3", config.engine_path.c_str(), "--model", config.model.c_str(), "--device", config.device.c_str(), "--verbosity",
             config.verbosity.c_str(), nullptr);
      std::cerr << "dino_backend: execlp failed: " << std::strerror(errno) << std::endl;
      _exit(127);
    }

    write_fd = to_child[1];
    read_fd = from_child[0];
    close(to_child[0]);
    close(from_child[1]);
  }

  void await_ready() {
    std::lock_guard<std::mutex> lock(io_mutex);
    if (ready) {
      return;
    }
    send_request("GET /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n", {});
    const http_response response = read_response(read_fd);
    if (response.status != 200) {
      throw std::runtime_error("dino_backend: health check failed: " + std::to_string(response.status) + " " + response.body);
    }
    ready = true;
  }

  std::vector<dino_detection> run(const cv::Mat &image, const std::vector<std::string> &classes, float box_threshold, float text_threshold,
                                  int max_detections) {
    if (image.empty() || classes.empty()) {
      return {};
    }
    await_ready();

    std::vector<uchar> bmp;
    if (!cv::imencode(".bmp", image, bmp)) {
      throw std::runtime_error("dino_backend: cv::imencode(.bmp) failed");
    }

    const std::string prompt = prompt_from_classes(classes);
    if (prompt.empty()) {
      return {};
    }

    std::ostringstream path;
    path << "/run_dino?box_threshold=" << box_threshold << "&text_threshold=" << text_threshold << "&max_detections=" << max_detections << "#"
         << url_encode(prompt);

    std::ostringstream header;
    header << "POST " << path.str() << " HTTP/1.1\r\n"
           << "Content-Type: image/bmp\r\n"
           << "Content-Length: " << bmp.size() << "\r\n\r\n";

    std::lock_guard<std::mutex> lock(io_mutex);
    send_request(header.str(), bmp);
    const http_response response = read_response(read_fd);
    if (response.status != 200) {
      throw std::runtime_error("dino_backend: DINO request failed: " + std::to_string(response.status) + " " + response.body);
    }
    return parse_detections(response.body);
  }

  void shutdown() {
    if (write_fd >= 0 && read_fd >= 0) {
      try {
        std::lock_guard<std::mutex> lock(io_mutex);
        send_request("GET /shutdown HTTP/1.1\r\nContent-Length: 0\r\n\r\n", {});
        (void)read_response(read_fd);
      } catch (...) {
      }
    }
    if (write_fd >= 0) {
      close(write_fd);
      write_fd = -1;
    }
    if (read_fd >= 0) {
      close(read_fd);
      read_fd = -1;
    }
    if (pid > 0) {
      int status = 0;
      waitpid(pid, &status, 0);
      pid = -1;
    }
  }

  void send_request(const std::string &header, const std::vector<uchar> &body) {
    write_all(write_fd, header.data(), header.size());
    if (!body.empty()) {
      write_all(write_fd, body.data(), body.size());
    }
  }

  dino_config config;
  bool lazy_init = true;
  bool ready = false;
  pid_t pid = -1;
  int write_fd = -1;
  int read_fd = -1;
  std::mutex io_mutex;
};

dino_backend::dino_backend(dino_config config, bool lazy_init) { impl_.reset(new impl(std::move(config), lazy_init)); }

dino_backend::~dino_backend() = default;

dino_backend::dino_backend(dino_backend &&) noexcept = default;
dino_backend &dino_backend::operator=(dino_backend &&) noexcept = default;

std::vector<dino_detection> dino_backend::run_dino(const cv::Mat &image, const std::vector<std::string> &classes, float box_threshold,
                                                   float text_threshold, int max_detections) {
  return impl_->run(image, classes, box_threshold, text_threshold, max_detections);
}

void dino_backend::await_ready() {
  impl_->await_ready();
}

bool dino_backend::is_ready() const {
  return impl_ && impl_->ready;
}

} // namespace ov_dino
