#include "target_generators_april_tag.hpp"

#include <gtest/gtest.h>

#include "april_tag_handling.hpp"
#include "generated_apriltag_code/tagCustom36h11.h"

using namespace reprojection_calibration::feature_extraction;

TEST(TargetGeneratorsAprilTag, TestGenerateAprilBoard) {
    cv::Size const pattern_size{4, 3};
    int const bit_size_pixel{10};
    AprilTagFamily const april_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    cv::Mat const april_board{GenerateAprilBoard(pattern_size, bit_size_pixel, april_family_handler.tag_family->codes)};

    EXPECT_EQ(april_board.rows, 420);
    EXPECT_EQ(april_board.cols, 560);
}

TEST(TargetGeneratorsAprilTag, TestGenerateAprilTag) {
    AprilTagFamily const april_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, bit_size_pixel)};

    EXPECT_EQ(april_tag.rows, 140);
    EXPECT_EQ(april_tag.cols, 140);
}

TEST(TargetGeneratorsAprilTag, TestCalculateCodeMatrix) {
    AprilTagFamily const april_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_family_handler.tag_family->codes[0])};

    // Check two properties of the matrix and hope if anything in the implementation breaks these catch it -_-
    EXPECT_EQ(code_matrix.sum(), 17);                                                               // Heuristic
    EXPECT_TRUE(code_matrix.row(5).isApprox(Eigen::Vector<int, 6>{1, 1, 1, 0, 0, 1}.transpose()));  // Heuristic
}