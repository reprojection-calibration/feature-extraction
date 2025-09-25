#include "target_generators_april_tag.hpp"

#include <gtest/gtest.h>

#include "april_tag_family_36h11.hpp"
#include "april_tag_family_custom_25h9.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(TargetGeneratorsAprilTag, TestGenerateAprilBoard) {
    cv::Size const pattern_size{4, 3};
    int const border_thickness_bits{2};  // Kalibr uses two, but most april grid applications only use one border bit.
    int const tag_spacing_bits{2};       // Kalibr defines tag spacing differently  as a fraction of the "metricSize"
    int const bit_size_pixel{10};
    cv::Mat const april_board{
        GenerateAprilBoard(pattern_size, border_thickness_bits, tag_spacing_bits, bit_size_pixel, april_tag::t36h11)};

    EXPECT_EQ(april_board.rows, 580);
    EXPECT_EQ(april_board.cols, 700);
}

TEST(TargetGeneratorsAprilTag, TestGenerateAprilTag) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(25, april_tag::custom25h9[35])};
    int const border_thickness_bits{2};  // Kalibr uses two, but most april grid applications only use one border bit.
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, border_thickness_bits, bit_size_pixel)};

    // TODO(Jack): Would it be better to calculate these values? Make sure we are consistent across all board types!
    EXPECT_EQ(april_tag.rows, 1010);
    EXPECT_EQ(april_tag.cols, 100);
}

TEST(TargetGeneratorsAprilTag, TestCalculateCodeMatrix) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_tag::t36h11[0])};

    // Check two properties of the matrix and hope if anything in the implementation breaks these catch it -_-
    EXPECT_EQ(code_matrix.sum(), 20);                                                               // Heuristic
    EXPECT_TRUE(code_matrix.row(5).isApprox(Eigen::Vector<int, 6>{1, 1, 1, 0, 1, 1}.transpose()));  // Heuristic
}