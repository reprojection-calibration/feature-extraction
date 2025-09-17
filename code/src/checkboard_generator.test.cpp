#include "checkboard_generator.hpp"

#include <gtest/gtest.h>

using namespace reprojection_calibration::feature_extraction;

TEST(CheckboardGenerator, GGG) {
    cv::Mat const checkboard_image{GenerateCheckboard(2, 4, 50)};

    EXPECT_FALSE(false);
}