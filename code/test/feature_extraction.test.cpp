#include "feature_extraction/feature_extraction.hpp"

#include <gtest/gtest.h>

#include "checkerboard_generator.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestExtractCheckerboardFeatures) {
    int const rows{2};
    int const cols{4};
    int const square_size{50};
    cv::Mat const image{GenerateCheckboard(rows, cols, square_size)};

    auto const [pixels, points]{ExtractCheckerboardFeatures(image, cv::Point{rows, cols})};

    EXPECT_FALSE(false);
}
