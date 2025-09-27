
#include <gtest/gtest.h>

#include "april_tag_cpp_wrapper.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include <apriltag/apriltag.h>

#include "generated_apriltag_code/tagCustom36h11.h"
}

// {}
// []

namespace reprojection_calibration::feature_extraction {

struct AprilTagDetection {
    AprilTagDetection(apriltag_detection_t const& raw_detection) {
        // Grab the homography
        using RowMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        Eigen::Map<RowMatrix3d> const H_map{raw_detection.H->data};
        H = H_map;

        // Grab the points
        for (int i{0}; i < 4; i++) {
            p.row(i) = Eigen::Vector2d{raw_detection.p[i][0], raw_detection.p[i][1]}.transpose();
        }
    }

    Eigen::Matrix3d H;
    Eigen::Matrix<double, 4, 2> p;
};

}  // namespace reprojection_calibration::feature_extraction

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractorsAprilTag, HHH) {
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetector const tag_detector{tag_family_handler, {2.0, 0.0, 1, false, false}};

    Eigen::MatrixXi const code_matrix{
        CalculateCodeMatrix(tag_family_handler.tag_family->nbits, tag_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(bit_size_pixel, code_matrix)};

    AprilTagDetections const raw_detections{tag_detector.Detect(april_tag)};
    AprilTagDetection const detection{raw_detections[0]};

    for (Eigen::Index i{0}; i < 4; ++i) {
        Eigen::Vector2d const p{detection.p.row(i)};
        cv::circle(april_tag, cv::Point(p(0), p(1)), 1, cv::Scalar(127), 1, cv::LINE_8);
    }

    EXPECT_EQ(1, 2);
}
