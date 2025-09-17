#include "checkerboard_generator.hpp"

#include <gtest/gtest.h>

using namespace reprojection_calibration::feature_extraction;

TEST(CheckerboardGenerator, TestGenerateCheckboard) {
    int const rows{3};
    int const cols{4};
    int const square_size{50};
    cv::Mat const checkerboard_image{GenerateCheckboard(rows, cols, square_size)};

    int const buffer{2 * square_size};
    EXPECT_EQ(checkerboard_image.rows, buffer + (rows + 1) * square_size);
    EXPECT_EQ(checkerboard_image.cols, buffer + (cols + 1) * square_size);
}