#include <gtest/gtest.h>

#include "feature_extraction/target_extractors.hpp"
#include "target_generators.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestExtractCheckerboardFeatures) {
    // WARN(Jack): Must be bigger than two for the checkerboard extraction (opencv error)
    int const rows{3};
    int const cols{4};
    int const unit_dimension_pixels{50};
    cv::Mat const image{GenerateCheckerboard(rows, cols, unit_dimension_pixels)};

    cv::Size const dimension{rows, cols};
    double const unit_dimension_meters{0.05};
    auto const checkerboard_extractor{std::make_unique<CheckerboardExtractor>(dimension, unit_dimension_meters)};

    auto const [pixels, points]{checkerboard_extractor->ExtractTarget(image)};

    EXPECT_EQ(pixels.rows(), rows * cols);
    EXPECT_TRUE(pixels.row(0).isApprox(Eigen::Vector2d{250, 100}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels.row(11).isApprox(Eigen::Vector2d{100, 200}.transpose(), 1e-6));  // Last pixel - heuristic
}
