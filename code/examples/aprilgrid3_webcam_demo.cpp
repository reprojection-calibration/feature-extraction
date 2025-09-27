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
int main() {
    cv::VideoCapture cap(0);  // , cv::CAP_V4L2
    if (not cap.isOpened()) {
        std::cerr << "Couldn't open video capture device" << std::endl;
        return -1;
    }

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::imshow("Tag Detections", frame);
        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}