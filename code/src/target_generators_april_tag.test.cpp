#include "target_generators_april_tag.hpp"

#include <gtest/gtest.h>

#include "april_tag_family_36h11.hpp"
#include "april_tag_family_custom_25h9.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(TargetGeneratorsAprilTag, TestGenerateAprilBoard) {
    cv::Size const pattern_size{4, 3};
    int const bit_size_pixel{10};
    cv::Mat const april_board{
        GenerateAprilBoard(pattern_size,  bit_size_pixel, april_tag::custom25h9)};

    EXPECT_EQ(april_board.rows, 390);
    EXPECT_EQ(april_board.cols, 520);
}

TEST(TargetGeneratorsAprilTag, TestGenerateAprilTag) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(25, april_tag::custom25h9[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, bit_size_pixel)};

    // TODO(Jack): Would it be better to calculate these values? Make sure we are consistent across all board types!
    EXPECT_EQ(april_tag.rows, 130);
    EXPECT_EQ(april_tag.cols, 130);
}

TEST(TargetGeneratorsAprilTag, TestCalculateCodeMatrix) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(25, april_tag::custom25h9[0])};

    // Check two properties of the matrix and hope if anything in the implementation breaks these catch it -_-
    EXPECT_EQ(code_matrix.sum(), 11);// Heuristic
    EXPECT_TRUE(code_matrix.row(4).isApprox(Eigen::Vector<int, 5>{0,0,0,1,1}.transpose()));  // Heuristic
}