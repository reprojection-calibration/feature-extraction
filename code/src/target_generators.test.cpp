#include <gtest/gtest.h>

#include "target_generators.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(TargetGenerators, TestGenerateCheckboard) {
    int const rows{3};
    int const cols{4};
    int const square_size{50};
    cv::Mat const checkerboard_image{GenerateCheckerboard(rows, cols, square_size)};

    int const border{2 * square_size};
    EXPECT_EQ(checkerboard_image.rows, border + (rows + 1) * square_size);
    EXPECT_EQ(checkerboard_image.cols, border + (cols + 1) * square_size);
}

TEST(TargetGenerators, TestGenerateCircleGrid) {
    int const rows{3};
    int const cols{4};
    int const circle_radius{25};
    int const circle_spacing{20};  // Between circle edges
    // TODO MAKE TRUE
    bool const asymmetric{false};
    cv::Mat const circlegrid_image{GenerateCircleGrid(rows, cols, circle_radius, circle_spacing, asymmetric)};

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