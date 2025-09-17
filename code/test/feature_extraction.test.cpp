#include "feature_extraction/feature_extraction.hpp"

#include <gtest/gtest.h>

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestCheckboardExtraction) {
    cv::Mat const image; // ADD IMAGE LOADING
    TargetType const target_type{TargetType::Chessboard};
    auto const [pixels, points]{ExtractFeatures(image, target_type)};

    EXPECT_FALSE(false);
}
