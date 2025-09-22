#include <gtest/gtest.h>

#include "checkerboard_extractor.hpp"
#include "feature_extraction/target_extractors.hpp"
#include "target_generators.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(CheckerboardExtractor, TestExtractCheckerboardFeatures) {
    int const rows{3};
    int const cols{4};
    int const unit_dimension_pixels{50};
    cv::Mat const image{GenerateCheckerboard(rows, cols, unit_dimension_pixels)};

    cv::Size const dimension{rows, cols};
    auto const pixels{CheckerboardExtractorExtractPixelFeatures(image, dimension)};

    EXPECT_EQ(pixels.rows(), rows * cols);
    EXPECT_TRUE(pixels.row(0).isApprox(Eigen::Vector2d{250, 100}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels.row(11).isApprox(Eigen::Vector2d{100, 200}.transpose(), 1e-6));  // Last pixel - heuristic

    double const unit_dimension_meters{0.05};
    auto const points{CheckerboardExtractorExtractPointFeatures(dimension, unit_dimension_meters)};
    EXPECT_EQ(points.rows(), rows * cols);
    EXPECT_TRUE(points.row(0).isApprox(Eigen::Vector3d{0, 0, 0}.transpose()));        // First point - heuristic
    EXPECT_TRUE(points.row(11).isApprox(Eigen::Vector3d{0.15, 0.1, 0}.transpose()));  // Last point - heuristic
}
