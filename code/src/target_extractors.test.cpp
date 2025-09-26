#include "target_extractors.hpp"

#include <gtest/gtest.h>

#include "target_generators.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractors, TestExtractCheckerboardFeatures) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const square_size_pixels{50};
    cv::Mat const image{GenerateCheckerboard(pattern_size, square_size_pixels)};

    std::optional<Eigen::MatrixX2d> const pixels{CheckerboardExtractorExtractPixelFeatures(image, pattern_size)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), pattern_size.height * pattern_size.width);
    EXPECT_TRUE(pixels->row(0).isApprox(Eigen::Vector2d{100, 100}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels->row(11).isApprox(Eigen::Vector2d{250, 200}.transpose(), 1e-6));  // Last pixel - heuristic
}

TEST(TargetExtractors, TestExtractCirclegridFeatures) {
    cv::Size const pattern_size{4, 3};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};  // Between circle edges
    bool const asymmetric{false};
    cv::Mat const image{GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    auto const pixels{CirclegridExtractorExtractPixelFeatures(image, pattern_size, asymmetric)};

    EXPECT_EQ(pixels->rows(), pattern_size.width * pattern_size.height);
    EXPECT_TRUE(pixels->row(0).isApprox(Eigen::Vector2d{265, 195}.transpose(), 1e-6));  // First pixel - heuristic
    EXPECT_TRUE(pixels->row(11).isApprox(Eigen::Vector2d{55, 55}.transpose(), 1e-6));   // Last pixel - heuristic
}

TEST(TargetExtractors, TestExtractCirclegridFeaturesAsymmetric) {
    // Refactor to use cv::Size
    // WARN(Jack): Must be even (rows)! See comment below.
    // WARN(Jack): Must be an odd number (cols) to prevent 180 degree rotation symmetry!
    // https://answers.opencv.org/question/96561/calibration-with-findcirclesgrid-trouble-with-pattern-widthheight/
    cv::Size const pattern_size{7, 6};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};
    bool const asymmetric{true};
    cv::Mat const image{GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    // WARN(Jack): Violation of principle of least surprise! They count every column but only half the rows (i.e. the
    // ones sticking out on the left side)
    cv::Size const dimension{pattern_size.height / 2, pattern_size.width};
    auto const pixels{CirclegridExtractorExtractPixelFeatures(image, dimension, true)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(),
              (pattern_size.width * pattern_size.height) / 2);  // NOTE(Jack): Divide by two due to asymmetry!
    EXPECT_TRUE(pixels->row(0).isApprox(Eigen::Vector2d{475, 55}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels->row(20).isApprox(Eigen::Vector2d{55, 335}.transpose(), 1e-6));  // Last pixel - heuristic
}