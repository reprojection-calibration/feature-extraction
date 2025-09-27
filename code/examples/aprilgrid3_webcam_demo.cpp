#include "../include/feature_extraction/april_tag_cpp_wrapper.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include <apriltag/apriltag.h>

#include "feature_extraction/generated_apriltag_code/tagCustom36h11.h"
}

#include <iostream>

// To get this working from CLion dev env I followed this link:
// https://medium.com/@steffen.stautmeister/how-to-build-and-run-opencv-and-pytorch-c-with-cuda-support-in-docker-in-clion-6f485155deb8
// After doing that my toolchain "Container Settings" were:
//      -e DISPLAY=:0.0 --entrypoint= -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --privileged --rm

using namespace reprojection_calibration::feature_extraction;

int main() {
    cv::VideoCapture cap(0);  // , cv::CAP_V4L2
    if (not cap.isOpened()) {
        std::cerr << "Couldn't open video capture device" << std::endl;
        return -1;
    }

    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetector const tag_detector{tag_family_handler, {2.0, 0.0, 1, false, false}};

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<AprilTagDetection> const detections{tag_detector.Detect(gray)};
        for (auto const& detection : detections) {
            for (int i{0}; i < 4; ++i) {
                cv::circle(frame, cv::Point(detection.p.row(i)[0], detection.p.row(i)[1]), 1, cv::Scalar(255, 0, 0), 5,
                           cv::LINE_8);
            }
            Eigen::Matrix<double, 4, 2> const extraction_corners{EstimateExtractionCorners(detection.H)};
            for (int i{0}; i < 4; ++i) {
                cv::circle(frame, cv::Point(extraction_corners.row(i)[0], extraction_corners.row(i)[1]), 1, cv::Scalar(0, 0, 255), 5,
                           cv::LINE_8);
            }
            Eigen::Matrix<double, 4, 2> const refined_extraction_corners{
                RefineExtractionCorners(gray, extraction_corners)};

            for (int i{0}; i < 4; ++i) {
                cv::circle(frame, cv::Point(refined_extraction_corners.row(i)[0], refined_extraction_corners.row(i)[1]), 1, cv::Scalar(0, 255, 0), 5,
                           cv::LINE_8);
            }
        }

        cv::imshow("Tag Detections", frame);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}