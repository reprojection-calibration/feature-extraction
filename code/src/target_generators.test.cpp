#include "target_generators.hpp"

#include <gtest/gtest.h>

using namespace reprojection_calibration::feature_extraction;

TEST(TargetGenerators, TestGenerateCheckboard) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const square_size_pixels{50};
    cv::Mat const checkerboard_image{GenerateCheckerboard(pattern_size, square_size_pixels)};

    int const white_space_border{2 * square_size_pixels};
    EXPECT_EQ(checkerboard_image.rows, white_space_border + (pattern_size.height + 1) * square_size_pixels);
    EXPECT_EQ(checkerboard_image.cols, white_space_border + (pattern_size.width + 1) * square_size_pixels);
}

TEST(TargetGenerators, TestGenerateCircleGrid) {
    cv::Size const pattern_size{4, 3};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};  // Distance between circle rims - that means the straight line distance
                                          // between two circles in one row or columns will be (2*radius + spacing)
    bool const asymmetric{false};
    cv::Mat const circlegrid_image{
        GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    // ERROR(Jack): Calculate these values, do not hardcode them!!!
    EXPECT_EQ(circlegrid_image.rows, 250);
    EXPECT_EQ(circlegrid_image.cols, 320);
}

TEST(TargetGenerators, TestGenerateCircleGridAsymmetric) {
    cv::Size const pattern_size{4, 3};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};  // Between circle edges
    bool const asymmetric{true};
    cv::Mat const circlegrid_image{
        GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    // ERROR(Jack): Calculate these values, do not hardcode them!!!
    EXPECT_EQ(circlegrid_image.rows, 250);
    EXPECT_EQ(circlegrid_image.cols, 320);
}
