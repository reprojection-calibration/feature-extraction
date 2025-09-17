#include "feature_extraction/feature_extraction.hpp"

#include <gtest/gtest.h>

#include "checkerboard_generator.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestExtractCheckerboardFeatures) {
    // WARN(Jack): Must be bigger than two for the checkerboard extraction (opencv error)
    int const rows{3};
    int const cols{4};
    int const square_size{50};
    cv::Mat const image{GenerateCheckboard(rows, cols, square_size)};

    Eigen::MatrixX2d const pixels{ExtractCheckerboardFeatures(image, cv::Point{rows, cols})};

    EXPECT_EQ(pixels.rows(), rows * cols);
    EXPECT_TRUE(pixels.row(0).isApprox(Eigen::Vector2d{250, 100}.transpose(), 1e-6));  // First pixel - heuristic
    EXPECT_TRUE(pixels.row(11).isApprox(Eigen::Vector2d{100, 200}.transpose(), 1e-6));  // Last pixel - heuristic
}
