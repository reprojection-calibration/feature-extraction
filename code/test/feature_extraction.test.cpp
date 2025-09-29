
#include "feature_extraction/feature_extraction.hpp"

#include <gtest/gtest.h>

#include "target_extractors.hpp"
#include "target_generators.hpp"

namespace reprojection_calibration::feature_extraction {

// ERROR(Jack): The fact that we have to specify the grid type in the base class, and then also have subsidiary child
// classes for base classes smells to high heaven!!! Read the accepted answer here
// (https://stackoverflow.com/questions/307765/how-do-i-check-if-an-objects-type-is-a-particular-subclass-in-c) for some
// context. At this point I cannot confirm or deny the right way to move foward here, but once we have a working model
// we need to revisit this!
enum class GridType { Checkerboard, CircleGrid, AprilGrid3 };

struct GridConfig {
    GridType type;
    cv::Size pattern_size;
};

struct CircleGridConfig : GridConfig {
    bool asymmetric;
};

std::function<std::optional<Eigen::MatrixX2d>(cv::Mat const&)> BuildExtractor(
    GridConfig const* const extractor_config) {
    if (extractor_config->type == GridType::Checkerboard) {
        cv::Size const pattern_size{extractor_config->pattern_size};

        return [pattern_size](cv::Mat const& image) {
            return CheckerboardExtractorExtractPixelFeatures(image, pattern_size);
        };
    } else if (extractor_config->type == GridType::CircleGrid) {
        // SOME OF THE HACKIEST GARBAGE I EVER WROTE! This it the physical manifestation of the problem I outlined
        // above! That we are using inheritance wrong.
        // ERROR(Jack): Usage of const_cast<GridConfig*>
        CircleGridConfig const* const circle_grid_config{
            static_cast<CircleGridConfig*>(const_cast<GridConfig*>(extractor_config))};
        cv::Size const pattern_size{circle_grid_config->pattern_size};
        bool const asymmetric{circle_grid_config->asymmetric};

        return [pattern_size, asymmetric](cv::Mat const& image) {
            // WARN(Jack): See note in the circle grid target extractor test to understand the confusing nature of how
            // they specify the size of an asymmetric circle grid. We need to add this error handling directly and
            // enforced into the library.
            cv::Size const adjusted_size{pattern_size.height / 2, pattern_size.width};
            return CirclegridExtractorExtractPixelFeatures(image, adjusted_size, asymmetric);
        };
    } else {
        // DITTO ABOVE
        // ERROR(Jack): Usage of const_cast<GridConfig*>
        // AprilGrid3Config const* const april_grid3_config{
        //   static_cast<AprilGrid3Config*>(const_cast<GridConfig*>(extractor_config))};
        return [](cv::Mat const& image) {
            static_cast<void>(image);
            return Eigen::MatrixX2d{{0, 1}, {2, 3}, {4, 5}};
        };
    }
}

}  // namespace reprojection_calibration::feature_extraction

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestBuildExtractorCheckerboard) {
    cv::Size const pattern_size{4, 3};
    int const square_size_pixels{50};
    cv::Mat const image{GenerateCheckerboard(pattern_size, square_size_pixels)};

    GridConfig const checkerboard_config{GridType::Checkerboard, pattern_size};
    auto const extractor{BuildExtractor(&checkerboard_config)};
    std::optional<Eigen::MatrixX2d> const pixels{extractor(image)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), pattern_size.width * pattern_size.height);
}

TEST(FeatureExtraction, TestBuildExtractorCircleGrid) {
    cv::Size const pattern_size{7, 6};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};
    bool const asymmetric{true};
    cv::Mat const image{GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    CircleGridConfig const circle_grid_config{GridType::CircleGrid, pattern_size, asymmetric};
    auto const extractor{BuildExtractor(&circle_grid_config)};
    std::optional<Eigen::MatrixX2d> const pixels{extractor(image)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), (pattern_size.width * pattern_size.height) / 2);
}
