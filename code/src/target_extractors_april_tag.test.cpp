
#include <gtest/gtest.h>

#include "april_tag_handling.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include <apriltag/apriltag.h>

#include "generated_apriltag_code/tagCustom36h11.h"
}

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractorsAprilTag, TestAprilTagDetectorDetectAprilBoard) {
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetector const tag_detector{tag_family_handler, {2.0, 0.0, 1, false, false}};

    cv::Size const pattern_size{4, 3};
    int const bit_size_pixel{10};
    cv::Mat const april_board{GenerateAprilBoard(pattern_size, bit_size_pixel, tag_family_handler.tag_family->nbits,
                                                 tag_family_handler.tag_family->codes)};

    AprilTagDetections const detections{tag_detector.Detect(april_board)};
    EXPECT_EQ(detections.detections->size, pattern_size.height * pattern_size.width);
}

TEST(TargetExtractorsAprilTag, TestAprilTagDetectorDetectAprilTag) {
    // Setup detector
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetector const tag_detector{tag_family_handler, {2.0, 0.0, 1, false, false}};

    // Setup tag
    Eigen::MatrixXi const code_matrix{
        CalculateCodeMatrix(tag_family_handler.tag_family->nbits, tag_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, bit_size_pixel)};

    // Act
    AprilTagDetections const detections{tag_detector.Detect(april_tag)};

    // Test the detection
    apriltag_detection_t const detection_0{detections[0]};
    EXPECT_EQ(detections.detections->size, 1);
    EXPECT_EQ(detection_0.id, 0);
    // Center point
    EXPECT_FLOAT_EQ(detection_0.c[0], 71);
    EXPECT_FLOAT_EQ(detection_0.c[1], 71);

    // Point zero
    EXPECT_FLOAT_EQ(detection_0.p[0][0], 30);
    EXPECT_FLOAT_EQ(detection_0.p[0][1], 112);

    // Point one
    EXPECT_FLOAT_EQ(detection_0.p[1][0], 112);
    EXPECT_FLOAT_EQ(detection_0.p[1][1], 112);

    // Point two
    EXPECT_FLOAT_EQ(detection_0.p[2][0], 112);
    EXPECT_FLOAT_EQ(detection_0.p[2][1], 30);

    // Point three
    EXPECT_FLOAT_EQ(detection_0.p[3][0], 30);
    EXPECT_FLOAT_EQ(detection_0.p[3][1], 30);
}
