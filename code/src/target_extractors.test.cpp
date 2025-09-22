#include "target_extractors.hpp"

#include <gtest/gtest.h>

#include "target_extractors.hpp"
#include "target_generators.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(CheckerboardExtractor, TestExtractCheckerboardFeatures) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const unit_dimension_pixels{50};
    cv::Mat const image{GenerateCheckerboard(pattern_size, unit_dimension_pixels)};

    std::optional<Eigen::MatrixX2d> const pixels{CheckerboardExtractorExtractPixelFeatures(image, pattern_size)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), pattern_size.height * pattern_size.width);
    EXPECT_TRUE(pixels->row(0).isApprox(Eigen::Vector2d{100, 100}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels->row(11).isApprox(Eigen::Vector2d{250, 200}.transpose(), 1e-6));  // Last pixel - heuristic
}

TEST(CheckerboardExtractor, TestExtractCirclegridFeatures) {
    int const rows{3};
    int const cols{4};
    int const circle_radius{25};
    int const circle_spacing{20};  // Between circle edges
    bool const asymmetric{false};
    cv::Mat const image{GenerateCircleGrid(rows, cols, circle_radius, circle_spacing, asymmetric)};

    cv::Size const dimension{rows, cols};
    auto const pixels{CirclegridExtractorExtractPixelFeatures(image, dimension, asymmetric)};

    EXPECT_EQ(pixels->rows(), rows * cols);
    EXPECT_TRUE(pixels->row(0).isApprox(Eigen::Vector2d{55, 195}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels->row(11).isApprox(Eigen::Vector2d{265, 55}.transpose(), 1e-6));  // Last pixel - heuristic
}

TEST(CheckerboardExtractor, TestExtractCirclegridFeaturesAsymmetric) {
    // Refactor to use cv::Size
    // WARN(Jack): Must be even (rows)! See comment below.
    // WARN(Jack): Must be an odd number (cols) to prevent 180 degree rotation symmetry!
    // https://answers.opencv.org/question/96561/calibration-with-findcirclesgrid-trouble-with-pattern-widthheight/
    int const rows{6};
    int const cols{7};
    int const circle_radius{25};
    int const circle_spacing{20};
    bool const asymmetric{true};
    cv::Mat const image{GenerateCircleGrid(rows, cols, circle_radius, circle_spacing, asymmetric)};

    // WARN(Jack): Violation of principle of least surprise! They count every column but only half the rows (i.e. the
    // ones sticking out on the left side)
    cv::Size const dimension{rows / 2, cols};
    auto const pixels{CirclegridExtractorExtractPixelFeatures(image, dimension, true)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), (rows * cols) / 2);  // WARN(Jack): Divide by two due to asymmetry!
    EXPECT_TRUE(pixels->row(0).isApprox(Eigen::Vector2d{475, 55}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels->row(20).isApprox(Eigen::Vector2d{55, 335}.transpose(), 1e-6));  // Last pixel - heuristic
}