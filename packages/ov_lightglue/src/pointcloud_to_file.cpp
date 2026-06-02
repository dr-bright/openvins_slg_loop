/*
 * OpenVINS pointcloud recorder (local copy in ov_lightglue).
 * Streams PointCloud2 messages into a binary PCD preserving the incoming fields.
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>

#include <ros/ros.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/PointCloud2.h>

#include "utils/print.h"

namespace ov_lightglue {

class PointCloudRecorder {
public:
  explicit PointCloudRecorder(const std::string &filename) : filename_(filename) {
    boost::filesystem::path file_path(filename.c_str());
    if (!file_path.parent_path().empty() && boost::filesystem::create_directories(file_path.parent_path())) {
      ROS_INFO("Created folder path to output file.");
      ROS_INFO("Path: %s", file_path.parent_path().c_str());
    }
    if (boost::filesystem::exists(filename)) {
      ROS_WARN("Output file exists, deleting old file....");
      boost::filesystem::remove(filename);
    }
  }

  ~PointCloudRecorder() { close(); }

  void callback_pointcloud(const sensor_msgs::PointCloud2ConstPtr &msg) {
    if (msg == nullptr || msg->width * msg->height == 0) {
      return;
    }

    if (!schema_initialized_) {
      initialize_schema(*msg);
    } else if (!schema_matches(*msg)) {
      ROS_ERROR("Skipping PointCloud2 with schema different from the first message. PCD requires a single fixed field layout.");
      return;
    }

    const size_t point_count = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    std::vector<uint8_t> message_data;
    message_data.reserve(point_count * msg->point_step);
    for (uint32_t row = 0; row < msg->height; ++row) {
      const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(msg->row_step);
      for (uint32_t col = 0; col < msg->width; ++col) {
        const size_t point_offset = row_offset + static_cast<size_t>(col) * static_cast<size_t>(msg->point_step);
        if (point_offset + msg->point_step > msg->data.size()) {
          ROS_ERROR("Skipping malformed PointCloud2 message with data shorter than its declared dimensions.");
          return;
        }
        message_data.insert(message_data.end(), msg->data.begin() + point_offset, msg->data.begin() + point_offset + msg->point_step);
      }
    }

    if (!outfile_.is_open()) {
      open_file();
    }
    outfile_.seekp(0, std::ios::end);
    outfile_.write(reinterpret_cast<const char *>(message_data.data()), static_cast<std::streamsize>(message_data.size()));
    if (!outfile_.good()) {
      ROS_ERROR("Failed while appending point data to PCD file: %s", filename_.c_str());
      std::exit(EXIT_FAILURE);
    }

    cloud_.width += static_cast<uint32_t>(point_count);
    cloud_.row_step = cloud_.width * cloud_.point_step;
    cloud_.is_dense = cloud_.is_dense && msg->is_dense;
    received_messages_++;
    ROS_INFO("Appended %zu points to %s (total=%u)", point_count, filename_.c_str(), cloud_.width);
  }

private:
  static constexpr size_t COUNT_FIELD_WIDTH = 20;

  void initialize_schema(const sensor_msgs::PointCloud2 &msg) {
    cloud_.header.frame_id = msg.header.frame_id;
    cloud_.height = 1;
    cloud_.width = 0;
    cloud_.is_bigendian = msg.is_bigendian;
    cloud_.is_dense = msg.is_dense;
    cloud_.point_step = msg.point_step;
    cloud_.row_step = 0;
    cloud_.fields.clear();
    cloud_.fields.reserve(msg.fields.size());
    for (const sensor_msgs::PointField &field : msg.fields) {
      pcl::PCLPointField pcl_field;
      pcl_field.name = field.name;
      pcl_field.offset = field.offset;
      pcl_field.datatype = field.datatype;
      pcl_field.count = field.count;
      cloud_.fields.push_back(pcl_field);
    }
    schema_initialized_ = true;
  }

  bool schema_matches(const sensor_msgs::PointCloud2 &msg) const {
    if (msg.is_bigendian != cloud_.is_bigendian || msg.point_step != cloud_.point_step || msg.fields.size() != cloud_.fields.size()) {
      return false;
    }
    for (size_t i = 0; i < msg.fields.size(); ++i) {
      const sensor_msgs::PointField &field = msg.fields.at(i);
      const pcl::PCLPointField &stored = cloud_.fields.at(i);
      if (field.name != stored.name || field.offset != stored.offset || field.datatype != stored.datatype || field.count != stored.count) {
        return false;
      }
    }
    return true;
  }

  static bool pcd_field_info(uint8_t datatype, uint32_t &size, char &type) {
    switch (datatype) {
    case sensor_msgs::PointField::INT8:
      size = 1;
      type = 'I';
      return true;
    case sensor_msgs::PointField::UINT8:
      size = 1;
      type = 'U';
      return true;
    case sensor_msgs::PointField::INT16:
      size = 2;
      type = 'I';
      return true;
    case sensor_msgs::PointField::UINT16:
      size = 2;
      type = 'U';
      return true;
    case sensor_msgs::PointField::INT32:
      size = 4;
      type = 'I';
      return true;
    case sensor_msgs::PointField::UINT32:
      size = 4;
      type = 'U';
      return true;
    case sensor_msgs::PointField::FLOAT32:
      size = 4;
      type = 'F';
      return true;
    case sensor_msgs::PointField::FLOAT64:
      size = 8;
      type = 'F';
      return true;
    default:
      return false;
    }
  }

  static std::string format_count(uint32_t value) {
    const std::string digits = std::to_string(value);
    if (digits.size() > COUNT_FIELD_WIDTH) {
      ROS_ERROR("PCD point count is too large for reserved header slot.");
      std::exit(EXIT_FAILURE);
    }
    return std::string(COUNT_FIELD_WIDTH - digits.size(), ' ') + digits;
  }

  static void append_count_line(std::string &header, const std::string &name, size_t &value_offset, uint32_t value) {
    header += name;
    header += " ";
    value_offset = header.size();
    header += format_count(value);
    header += "\n";
  }

  std::string build_header() {
    std::ostringstream fields;
    std::ostringstream size;
    std::ostringstream type;
    std::ostringstream count;
    for (size_t i = 0; i < cloud_.fields.size(); ++i) {
      const pcl::PCLPointField &field = cloud_.fields.at(i);
      uint32_t field_size = 0;
      char field_type = '\0';
      if (!pcd_field_info(field.datatype, field_size, field_type)) {
        ROS_ERROR("Unsupported PointCloud2 datatype %u for field '%s'", field.datatype, field.name.c_str());
        std::exit(EXIT_FAILURE);
      }
      if (i > 0) {
        fields << " ";
        size << " ";
        type << " ";
        count << " ";
      }
      fields << field.name;
      size << field_size;
      type << field_type;
      count << field.count;
    }

    std::string header;
    header += "# .PCD v0.7 - Point Cloud Data file format\n";
    header += "VERSION 0.7\n";
    header += "FIELDS " + fields.str() + "\n";
    header += "SIZE " + size.str() + "\n";
    header += "TYPE " + type.str() + "\n";
    header += "COUNT " + count.str() + "\n";
    append_count_line(header, "WIDTH", width_value_offset_, cloud_.width);
    header += "HEIGHT 1\n";
    header += "VIEWPOINT 0 0 0 1 0 0 0\n";
    append_count_line(header, "POINTS", points_value_offset_, cloud_.width);
    header += "DATA binary\n";
    return header;
  }

  void open_file() {
    outfile_.open(filename_.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!outfile_.is_open()) {
      ROS_ERROR("Unable to open PCD file for writing: %s", filename_.c_str());
      std::exit(EXIT_FAILURE);
    }
    const std::string header = build_header();
    outfile_.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!outfile_.good()) {
      ROS_ERROR("Failed while writing initial PCD header: %s", filename_.c_str());
      std::exit(EXIT_FAILURE);
    }
    outfile_.seekp(0, std::ios::end);
  }

  void finalize_counts() {
    if (!outfile_.is_open()) {
      return;
    }
    const std::string count = format_count(cloud_.width);
    outfile_.seekp(static_cast<std::streamoff>(width_value_offset_), std::ios::beg);
    outfile_.write(count.data(), static_cast<std::streamsize>(count.size()));
    outfile_.seekp(static_cast<std::streamoff>(points_value_offset_), std::ios::beg);
    outfile_.write(count.data(), static_cast<std::streamsize>(count.size()));
    if (!outfile_.good()) {
      ROS_ERROR("Failed while finalizing PCD point counts: %s", filename_.c_str());
      std::exit(EXIT_FAILURE);
    }
    outfile_.flush();
    outfile_.seekp(0, std::ios::end);
  }

  void close() {
    if (closed_) {
      return;
    }
    closed_ = true;
    if (cloud_.width == 0) {
      ROS_WARN("No pointcloud points received, not writing PCD file.");
      return;
    }
    finalize_counts();
    if (outfile_.is_open()) {
      outfile_.close();
    }
    ROS_INFO("Closed %s after writing %u total points from %zu messages", filename_.c_str(), cloud_.width, received_messages_);
  }

  std::string filename_;
  pcl::PCLPointCloud2 cloud_;
  std::ofstream outfile_;
  size_t width_value_offset_ = 0;
  size_t points_value_offset_ = 0;
  size_t received_messages_ = 0;
  bool schema_initialized_ = false;
  bool closed_ = false;
};

} // namespace ov_lightglue

int main(int argc, char **argv) {
  ros::init(argc, argv, "pointcloud_to_file");
  ros::NodeHandle nh("~");

  std::string verbosity;
  nh.param<std::string>("verbosity", verbosity, "INFO");
  ov_core::Printer::setPrintLevel(verbosity);

  std::string topic, fileoutput;
  nh.getParam("topic", topic);
  nh.getParam("output", fileoutput);

  if (topic.empty() || fileoutput.empty()) {
    PRINT_ERROR("Required parameters: topic, output");
    return EXIT_FAILURE;
  }

  ov_lightglue::PointCloudRecorder recorder(fileoutput);
  ros::Subscriber sub = nh.subscribe(topic, 9999, &ov_lightglue::PointCloudRecorder::callback_pointcloud, &recorder);

  ros::spin();
  return EXIT_SUCCESS;
}
