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
    int const circle_size{50};
    // TODO MAKE NON ZERO!
    double const circle_spacing{0.0};  // Between circle centers? Oder wat?
    // TODO MAKE TRUE
    bool const asymmetric{false};
    cv::Mat const circlegrid_image{GenerateCircleGrid(rows, cols, circle_size, circle_spacing, asymmetric)};

    int const border{2 * circle_size};
    EXPECT_EQ(circlegrid_image.rows, border + (rows + 1) * circle_size);  // What about circle_spacing
    EXPECT_EQ(circlegrid_image.cols, border + (cols + 1) * circle_size);  // What about circle_spacing
}