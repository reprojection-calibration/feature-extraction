
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "april_tag_handling.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include "apriltag/apriltag.h"
#include "generated_apriltag_code/tagCustom36h11.h"
}

// {  }

namespace reprojection_calibration::feature_extraction {}  // namespace reprojection_calibration::feature_extraction

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractorsAprilTag, XXX) {
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetectorSettings const settings{2.0, 0.0, 1, false, false};
    AprilTagDetector const tag_detector{tag_family_handler, settings};

    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, tag_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, bit_size_pixel)};

    AprilTagDetections detections = tag_detector.Detect(april_tag);

    apriltag_detection_t const det{detections[0]};

    EXPECT_EQ(det.id, 0);
    EXPECT_EQ(det.c[0], 71);  // Roughly the center of tag 140 pixels wide
    EXPECT_EQ(det.c[1], 71);
}
