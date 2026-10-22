#include <ros/ros.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt8.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

class ProximityNode {
 public:
  ProximityNode() : private_("~") {
    private_.param<std::string>("input_topic", input_topic_, "/livox/lidar");
    private_.param("min_range", min_range_, 0.2);
    private_.param("max_range", max_range_, 5.0);
    private_.param("support_radius", radius_, 0.08);
    private_.param("min_neighbors", neighbors_, 3);
    private_.param("weak_distance", weak_, 2.0);
    private_.param("medium_distance", medium_, 1.0);
    private_.param("strong_distance", strong_, 0.5);
    private_.param("hysteresis", hysteresis_, 0.1);
    private_.param("release_frames", release_frames_, 2);
    private_.param("stale_timeout", timeout_, 0.5);
    if (!(std::isfinite(min_range_) && std::isfinite(max_range_) &&
          std::isfinite(radius_) && std::isfinite(weak_) && std::isfinite(medium_) &&
          std::isfinite(strong_) && std::isfinite(hysteresis_) && std::isfinite(timeout_) &&
          0 < min_range_ && min_range_ < strong_ && strong_ < medium_ &&
          medium_ < weak_ && weak_ < max_range_ && radius_ > 0 &&
          neighbors_ >= 1 && hysteresis_ >= 0 && timeout_ > 0 && release_frames_ >= 1)) {
      throw std::runtime_error("Invalid proximity parameters");
    }
    distance_pub_ = node_.advertise<std_msgs::Float32>("/haptic/proximity/distance", 1);
    level_pub_ = node_.advertise<std_msgs::UInt8>("/haptic/proximity/level", 1);
    state_pub_ = node_.advertise<std_msgs::String>("/haptic/proximity/state", 1);
    diagnostics_pub_ = node_.advertise<diagnostic_msgs::DiagnosticArray>("/haptic/proximity/diagnostics", 1);
    subscriber_ = node_.subscribe(input_topic_, 1, &ProximityNode::cloud, this);
    timer_ = node_.createWallTimer(ros::WallDuration(0.1), &ProximityNode::watchdog, this);
    ROS_WARN("PCL PROTOTYPE: all visible directions, no body/ground/tag filter, no motor control");
  }

 private:
  static double distance(const pcl::PointXYZ& p) {
    return std::hypot(std::hypot(double(p.x), double(p.y)), double(p.z));
  }
  int band(double d) const {
    return d <= strong_ ? 3 : d <= medium_ ? 2 : d <= weak_ ? 1 : 0;
  }
  double boundary(int level) const {
    return level == 3 ? strong_ : level == 2 ? medium_ : weak_;
  }
  int updateLevel(double d) {
    const int candidate = band(d);
    if (level_ < 0 || candidate >= level_) {
      level_ = candidate;
      release_count_ = 0;
    } else if (d > boundary(level_) + hysteresis_) {
      if (++release_count_ >= release_frames_) {
        level_ = candidate;
        release_count_ = 0;
      }
    } else {
      release_count_ = 0;
    }
    return level_;
  }
  static void value(diagnostic_msgs::DiagnosticStatus& status, const std::string& key,
                    const std::string& text) {
    diagnostic_msgs::KeyValue item;
    item.key = key;
    item.value = text;
    status.values.push_back(item);
  }
  void publish(double d, int level, const std::string& state,
               diagnostic_msgs::DiagnosticStatus status, const std_msgs::Header& header) {
    std_msgs::Float32 output;
    output.data = static_cast<float>(d);
    distance_pub_.publish(output);
    std_msgs::UInt8 strength;
    strength.data = level < 0 ? 255 : static_cast<uint8_t>(level);
    level_pub_.publish(strength);
    std_msgs::String text;
    text.data = state;
    state_pub_.publish(text);
    diagnostic_msgs::DiagnosticArray array;
    array.header = header;
    status.name = "haptic/proximity_prototype";
    status.hardware_id = "software_only";
    status.message = state;
    value(status, "distance_m", std::to_string(d));
    value(status, "level", level < 0 ? "UNKNOWN" : std::to_string(level));
    value(status, "input_topic", input_topic_);
    value(status, "motor_control", "disabled");
    value(status, "scope", "all_visible_directions_no_body_ground_or_tag_filter");
    array.status.push_back(status);
    diagnostics_pub_.publish(array);
  }
  void cloud(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    const auto started = ros::WallTime::now();
    received_ = true;
    last_received_ = started;
    stale_ = false;
    size_t zero = 0, invalid = 0, near = 0, far = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
    input->reserve(msg->points.size());
    double before = std::numeric_limits<double>::infinity();
    for (const auto& p : msg->points) {
      if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))) { ++invalid; continue; }
      if (p.x == 0 && p.y == 0 && p.z == 0) { ++zero; continue; }
      const double d = std::hypot(std::hypot(double(p.x), double(p.y)), double(p.z));
      if (d < min_range_) { ++near; continue; }
      if (d > max_range_) { ++far; continue; }
      input->push_back(pcl::PointXYZ(p.x, p.y, p.z));
      before = std::min(before, d);
    }
    pcl::PointCloud<pcl::PointXYZ> supported;
    if (!input->empty()) {
      pcl::RadiusOutlierRemoval<pcl::PointXYZ> filter;
      filter.setInputCloud(input);
      filter.setRadiusSearch(radius_);
      filter.setMinNeighborsInRadius(neighbors_);
      filter.filter(supported);
    }
    double nearest = std::numeric_limits<double>::infinity();
    for (const auto& p : supported) nearest = std::min(nearest, distance(p));
    diagnostic_msgs::DiagnosticStatus status;
    status.level = diagnostic_msgs::DiagnosticStatus::WARN;
    value(status, "input_points", std::to_string(msg->points.size()));
    value(status, "declared_count_matches", msg->point_num == msg->points.size() ? "true" : "false");
    value(status, "zero_points", std::to_string(zero));
    value(status, "nonfinite_points", std::to_string(invalid));
    value(status, "near_untrusted_points", std::to_string(near));
    value(status, "far_excluded_points", std::to_string(far));
    value(status, "range_points", std::to_string(input->size()));
    value(status, "supported_points", std::to_string(supported.size()));
    value(status, "removed_by_support", std::to_string(input->size() - supported.size()));
    value(status, "near_field_uncertain", near ? "true" : "false");
    value(status, "unsupported_closer_return", std::isfinite(before) && before + 1e-6 < nearest ? "true" : "false");
    value(status, "processing_ms", std::to_string((ros::WallTime::now()-started).toSec()*1000));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(nearest)) {
      level_ = -1;
      release_count_ = 0;
      publish(nan, -1, near ? "NEAR_FIELD_UNCERTAIN" : "NO_SUPPORTED_POINTS", status, msg->header);
      return;
    }
    const int level = updateLevel(nearest);
    const std::string name = level == 3 ? "STRONG" : level == 2 ? "MEDIUM" : level == 1 ? "WEAK" : "NO_NEAR_CANDIDATE";
    publish(nearest, level, near ? name + ";NEAR_FIELD_UNCERTAIN" : name, status, msg->header);
    ROS_INFO_THROTTLE(1.0, "%s: candidate %.3f m; points %zu -> %zu; near untrusted %zu",
                      name.c_str(), nearest, msg->points.size(), supported.size(), near);
  }
  void watchdog(const ros::WallTimerEvent&) {
    if (!received_ || (ros::WallTime::now()-last_received_).toSec() > timeout_) {
      level_ = -1;
      release_count_ = 0;
      diagnostic_msgs::DiagnosticStatus status;
      status.level = diagnostic_msgs::DiagnosticStatus::ERROR;
      value(status, "data_stale", "true");
      std_msgs::Header header;
      header.stamp = ros::Time::now();
      publish(std::numeric_limits<double>::quiet_NaN(), -1, "STALE_OR_NO_DATA", status, header);
      if (!stale_) ROS_WARN("Proximity input missing/stale; distance UNKNOWN, level 255");
      stale_ = true;
    }
  }
  ros::NodeHandle node_, private_;
  ros::Subscriber subscriber_;
  ros::Publisher distance_pub_, level_pub_, state_pub_, diagnostics_pub_;
  ros::WallTimer timer_;
  ros::WallTime last_received_;
  std::string input_topic_;
  double min_range_, max_range_, radius_, weak_, medium_, strong_, hysteresis_, timeout_;
  int neighbors_, release_frames_, level_ = -1, release_count_ = 0;
  bool received_ = false, stale_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "haptic_proximity_node");
  try {
    ProximityNode node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("%s", error.what());
    return 1;
  }
  return 0;
}
