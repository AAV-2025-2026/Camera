#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include "camera_decode/nvdec_decoder.hpp"
#include "camera_decode/yolo_detector.hpp"


class StopSignNode : public rclcpp::Node {
public:
    StopSignNode() : Node("stop_sign_node"), running_(true) {
        this->declare_parameter("camera_name",      "cam1");
        this->declare_parameter("rtsp_url",         "rtsp://192.168.1.102:8554/cam1");
        this->declare_parameter("stop_engine_path", "/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/best.engine");
        this->declare_parameter("tl_engine_path",   "/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/traffic_lights.engine");
        this->declare_parameter("pedestrian_engine_path", "/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/pedestrian.engine");
        this->declare_parameter("conf_thresh",      0.5);
        this->declare_parameter("tl_conf_thresh",   0.50);
        this->declare_parameter("pedestrian_conf_thresh", 0.5);
        this->declare_parameter("show_window",      false);

        auto camera_name  = this->get_parameter("camera_name").as_string();
        auto rtsp_url     = this->get_parameter("rtsp_url").as_string();
        auto stop_engine  = this->get_parameter("stop_engine_path").as_string();
        auto tl_engine    = this->get_parameter("tl_engine_path").as_string();
        auto pedestrian_engine = this->get_parameter("pedestrian_engine_path").as_string();
        float conf        = (float)this->get_parameter("conf_thresh").as_double();
        float tl_conf     = (float)this->get_parameter("tl_conf_thresh").as_double();
        float pedestrian_conf = (float)this->get_parameter("pedestrian_conf_thresh").as_double();
        show_window_      = this->get_parameter("show_window").as_bool();
        camera_name_      = camera_name;

        decoder_   = std::make_unique<camera_decode::NvDecDecoder>(rtsp_url);
        yolo_stop_ = std::make_unique<YoloDetector>(stop_engine, conf);
        yolo_tl_   = std::make_unique<YoloDetector>(tl_engine, tl_conf);
        yolo_pedestrian_ = std::make_unique<YoloDetector>(pedestrian_engine, pedestrian_conf);

        // Stop sign publishers
        stop_bool_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/aav/" + camera_name_ + "/stop_sign_detected", 10);
        stop_conf_pub_ = this->create_publisher<std_msgs::msg::Float32>(
            "/aav/" + camera_name_ + "/stop_sign_confidence", 10);
        stop_bbox_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "/aav/" + camera_name_ + "/stop_sign_detections", 10);

        // Traffic light publishers
        tl_bool_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/aav/" + camera_name_ + "/traffic_light_detected", 10);
        tl_conf_pub_ = this->create_publisher<std_msgs::msg::Float32>(
            "/aav/" + camera_name_ + "/traffic_light_confidence", 10);
        tl_bbox_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "/aav/" + camera_name_ + "/traffic_light_detections", 10);

        // Pedestrian publishers
        pedestrian_bool_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/aav/" + camera_name_ + "/pedestrian_detected", 10);
        pedestrian_conf_pub_ = this->create_publisher<std_msgs::msg::Float32>(
            "/aav/" + camera_name_ + "/pedestrian_confidence", 10);
        pedestrian_bbox_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "/aav/" + camera_name_ + "/pedestrian_detections", 10);
        
        // ── Thread 1: Decoder — always overwrites latest frame (Fix 3) ──
        decoder_thread_ = std::thread([this]() {
            while (running_) {
                cv::Mat f = decoder_->getFrame();
                if (!f.empty()) {
                    std::lock_guard<std::mutex> lock(frame_mtx_);
                    frame_ = f.clone();  // always latest, never queued
                }
            }
        });

        // ── Thread 2: YOLO inference — async, never blocks display (Fix 2) ──
        infer_thread_ = std::thread([this]() {
            while (running_) {
                cv::Mat f;
                {
                    std::lock_guard<std::mutex> lock(frame_mtx_);
                    if (frame_.empty()) continue;
                    f = frame_.clone();
                }

                // Run both detectors sequentially on same frame
                // Both share the same GPU so parallel threads won't help
                auto stop_dets = yolo_stop_->detect(f);
                auto tl_dets   = yolo_tl_->detect(f);
                auto pedestrian_dets = yolo_pedestrian_->detect(f);
                {
                    std::lock_guard<std::mutex> lock(det_mtx_);
                    latest_stop_dets_ = stop_dets;
                    latest_tl_dets_   = tl_dets;
                    latest_pedestrian_dets_ = pedestrian_dets;
                    infer_frame_      = f;  // frame that detections correspond to
                }
            }
        });

        // ── Timer: display + publish only — never waits for inference ────
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            [this]() { timerCallback(); });

        RCLCPP_INFO(this->get_logger(), "Yolo Detector (window=%s)",
                    show_window_ ? "ON" : "OFF");
    }

    ~StopSignNode() {
        running_ = false;
        if (decoder_thread_.joinable()) decoder_thread_.join();
        if (infer_thread_.joinable())   infer_thread_.join();
        if (show_window_) cv::destroyAllWindows();
    }

private:

    std::string camera_name_;

    void publishDetections(
        cv::Mat& frame,
        const std::vector<Detection>& dets,
        const std::string& class_label,
        const cv::Scalar& color,
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr& bool_pub,
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr& conf_pub,
        rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr& bbox_pub)
    {
        bool  found     = !dets.empty();
        float best_conf = 0.0f;

        // ── Build Detection2DArray ────────────────────────────
        vision_msgs::msg::Detection2DArray bbox_msg;
        bbox_msg.header.stamp    = this->now();
        bbox_msg.header.frame_id = camera_name_;

        for (auto& d : dets) {
            if (d.conf > best_conf) best_conf = d.conf;

            vision_msgs::msg::Detection2D det;
            det.bbox.center.position.x = d.bbox.x + d.bbox.width  / 2.0;
            det.bbox.center.position.y = d.bbox.y + d.bbox.height / 2.0;
            det.bbox.size_x            = d.bbox.width;
            det.bbox.size_y            = d.bbox.height;

            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = class_label;
            hyp.hypothesis.score    = d.conf;
            det.results.push_back(hyp);
            bbox_msg.detections.push_back(det);

            if (show_window_) {
                cv::rectangle(frame, d.bbox, color, 2);
                std::string lbl = cv::format("%s %.0f%%", class_label.c_str(), d.conf * 100);
                cv::putText(frame, lbl,
                            {d.bbox.x, d.bbox.y - 8},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }
        }

        // ── Publish ───────────────────────────────────────────
        std_msgs::msg::Bool bool_msg;
        bool_msg.data = found;
        bool_pub->publish(bool_msg);

        std_msgs::msg::Float32 conf_msg;
        conf_msg.data = best_conf;
        conf_pub->publish(conf_msg);

        bbox_pub->publish(bbox_msg);

        if (found)
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                 "%s detected (conf=%.0f%%)", class_label.c_str(), best_conf * 100);
    }

    void timerCallback() {
        cv::Mat frame;
        std::vector<Detection> stop_dets, tl_dets, pedestrian_dets;

        // Grab latest inferred frame + detections — non-blocking
        {
            std::lock_guard<std::mutex> lock(det_mtx_);
            if (infer_frame_.empty()) return;
            frame     = infer_frame_.clone();
            stop_dets = latest_stop_dets_;
            tl_dets   = latest_tl_dets_;
            pedestrian_dets = latest_pedestrian_dets_;
        }

        publishDetections(frame, stop_dets, "stop_sign",     {0, 0, 255},
                          stop_bool_pub_, stop_conf_pub_, stop_bbox_pub_);

        publishDetections(frame, tl_dets,   "traffic_light", {0, 255, 0},
                          tl_bool_pub_,   tl_conf_pub_,   tl_bbox_pub_);

        publishDetections(frame, pedestrian_dets, "pedestrian", {255, 0, 0},
                          pedestrian_bool_pub_, pedestrian_conf_pub_, pedestrian_bbox_pub_);
        
        // ── Display ───────────────────────────────────────────
        if (show_window_) {
            cv::imshow("Yolo Detector", frame);
            if (cv::waitKey(1) == 'q') rclcpp::shutdown();
        }
    }

    // Decoder
    std::unique_ptr<camera_decode::NvDecDecoder> decoder_;
    std::unique_ptr<YoloDetector>                yolo_stop_;
    std::unique_ptr<YoloDetector>                yolo_tl_;
    std::unique_ptr<YoloDetector>                yolo_pedestrian_;
    // Publishers — stop sign
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                stop_bool_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr             stop_conf_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr stop_bbox_pub_;

    // Publishers — traffic light
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                tl_bool_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr             tl_conf_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr tl_bbox_pub_;

    // Publishers — pedestrian
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                pedestrian_bool_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr             pedestrian_conf_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pedestrian_bbox_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    // Thread 1 — decoder
    std::thread decoder_thread_;
    std::mutex  frame_mtx_;
    cv::Mat     frame_;

    // Thread 2 — inference
    std::thread            infer_thread_;
    std::mutex             det_mtx_;
    cv::Mat                infer_frame_;
    std::vector<Detection> latest_stop_dets_;
    std::vector<Detection> latest_tl_dets_;
    std::vector<Detection> latest_pedestrian_dets_;
    std::atomic<bool> running_;
    bool show_window_ = false;
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StopSignNode>());
    rclcpp::shutdown();
    return 0;
}