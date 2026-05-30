/*
 * OpenVINS pointcloud recorder (local copy in ov_lightglue).
 * Accumulates PointCloud2 messages and writes a PCD with x y z featid lifetime timestamp.
 */

#include <cstdint>
#include <cstring>
#include <string>

#include <boost/filesystem.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

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

    cloud_.header.frame_id = "global";
    cloud_.height = 1;
    cloud_.width = 0;
    cloud_.is_bigendian = false;
    cloud_.is_dense = false;
    cloud_.point_step = POINT_STEP;
    cloud_.row_step = 0;
    cloud_.fields.resize(6);
    set_field(0, "x", 0, pcl::PCLPointField::FLOAT32);
    set_field(1, "y", 4, pcl::PCLPointField::FLOAT32);
    set_field(2, "z", 8, pcl::PCLPointField::FLOAT32);
    set_field(3, "featid", 12, pcl::PCLPointField::UINT32);
    set_field(4, "lifetime", 16, pcl::PCLPointField::UINT32);
    set_field(5, "timestamp", 20, pcl::PCLPointField::FLOAT64);
  }

  ~PointCloudRecorder() { write(); }

  void callback_pointcloud(const sensor_msgs::PointCloud2ConstPtr &msg) {
    if (msg == nullptr || msg->width * msg->height == 0) {
      return;
    }

    const double timestamp = msg->header.stamp.toSec();
    cloud_.header.frame_id = msg->header.frame_id.empty() ? cloud_.header.frame_id : msg->header.frame_id;

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
    const bool has_feat_id = has_field(*msg, "feat_id") || has_field(*msg, "featid");
    const bool has_lifetime = has_field(*msg, "lifetime");

    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<uint32_t>> it_feat_id;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<uint32_t>> it_lifetime;
    if (has_feat_id) {
      it_feat_id.reset(new sensor_msgs::PointCloud2ConstIterator<uint32_t>(*msg, has_field(*msg, "feat_id") ? "feat_id" : "featid"));
    }
    if (has_lifetime) {
      it_lifetime.reset(new sensor_msgs::PointCloud2ConstIterator<uint32_t>(*msg, "lifetime"));
    }

    const size_t point_count = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    cloud_.data.reserve(cloud_.data.size() + point_count * POINT_STEP);
    for (size_t i = 0; i < point_count; ++i, ++it_x, ++it_y, ++it_z) {
      const float x = *it_x;
      const float y = *it_y;
      const float z = *it_z;
      const uint32_t feat_id = has_feat_id ? **it_feat_id : 0;
      const uint32_t lifetime = has_lifetime ? **it_lifetime : 0;

      append_value(x);
      append_value(y);
      append_value(z);
      append_value(feat_id);
      append_value(lifetime);
      append_value(timestamp);

      if (has_feat_id) {
        ++(*it_feat_id);
      }
      if (has_lifetime) {
        ++(*it_lifetime);
      }
    }
    cloud_.width += static_cast<uint32_t>(point_count);
    cloud_.row_step = cloud_.width * cloud_.point_step;
    received_messages_++;
  }

private:
  static constexpr uint32_t POINT_STEP = 28;

  void set_field(size_t index, const std::string &name, uint32_t offset, uint8_t datatype) {
    cloud_.fields.at(index).name = name;
    cloud_.fields.at(index).offset = offset;
    cloud_.fields.at(index).datatype = datatype;
    cloud_.fields.at(index).count = 1;
  }

  static bool has_field(const sensor_msgs::PointCloud2 &msg, const std::string &name) {
    for (const sensor_msgs::PointField &field : msg.fields) {
      if (field.name == name) {
        return true;
      }
    }
    return false;
  }

  template <typename T> void append_value(const T &value) {
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&value);
    cloud_.data.insert(cloud_.data.end(), bytes, bytes + sizeof(T));
  }

  void write() {
    if (wrote_) {
      return;
    }
    wrote_ = true;
    if (cloud_.width == 0) {
      ROS_WARN("No pointcloud points received, not writing PCD file.");
      return;
    }
    pcl::PCDWriter writer;
    const int rc = writer.writeBinary(filename_, cloud_);
    if (rc != 0) {
      ROS_ERROR("Failed to write PCD file: %s", filename_.c_str());
      return;
    }
    ROS_INFO("Wrote %u points from %zu messages to %s", cloud_.width, received_messages_, filename_.c_str());
  }

  std::string filename_;
  pcl::PCLPointCloud2 cloud_;
  size_t received_messages_ = 0;
  bool wrote_ = false;
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
