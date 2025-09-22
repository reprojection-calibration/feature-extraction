#include "target_generators.hpp"

#include <gtest/gtest.h>

using namespace reprojection_calibration::feature_extraction;

TEST(TargetGenerators, TestGenerateCheckboard) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const square_size{50};
    cv::Mat const checkerboard_image{GenerateCheckerboard(pattern_size, square_size)};

    int const border{2 * square_size};
    EXPECT_EQ(checkerboard_image.rows, border + (pattern_size.height + 1) * square_size);
    EXPECT_EQ(checkerboard_image.cols, border + (pattern_size.width + 1) * square_size);
}

TEST(TargetGenerators, TestGenerateCircleGrid) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const circle_radius{25};
    int const circle_spacing{20};  // Between circle edges
    bool const asymmetric{false};
    cv::Mat const circlegrid_image{GenerateCircleGrid(pattern_size, circle_radius, circle_spacing, asymmetric)};

    // ERROR(Jack): Calculate these values, do not hardcode them!!!
    EXPECT_EQ(circlegrid_image.rows, 250);  // What about circle_spacing
    EXPECT_EQ(circlegrid_image.cols, 320);  // What about circle_spacing
}

TEST(TargetGenerators, TestGenerateCircleGridAsymmetric) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const circle_radius{25};
    int const circle_spacing{20};  // Between circle edges
    bool const asymmetric{true};
    cv::Mat const circlegrid_image{GenerateCircleGrid(pattern_size, circle_radius, circle_spacing, asymmetric)};

    // ERROR(Jack): Calculate these values, do not hardcode them!!!
    EXPECT_EQ(circlegrid_image.rows, 250);  // What about circle_spacing
    EXPECT_EQ(circlegrid_image.cols, 320);  // What about circle_spacing
}

TEST(TestGenerators, TestGenerateGridIndices) {
    int const rows{3};
    int const cols{4};

    Eigen::ArrayX2i const grid_indices{GenerateGridIndices(rows, cols)};

    EXPECT_EQ(grid_indices.rows(), rows * cols);
    // Heuristically check the first grid row that it is ((0,0), (0,1), (0,2), (0,3))
    EXPECT_TRUE(grid_indices.col(0).topRows(cols).isApprox(Eigen::ArrayXi::Zero(cols)));
    EXPECT_TRUE(grid_indices.col(1).topRows(cols).isApprox(Eigen::ArrayXi::LinSpaced(cols, 0, cols)));
}