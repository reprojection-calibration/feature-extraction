
#include "feature_extraction/feature_extraction.hpp"

#include <gtest/gtest.h>

#include "target_extractors.hpp"
#include "target_generators.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Put these implementations of the pure virtual base class into the src folder, these are not part of the
// public interface!

class CheckerboardExtractor : public TargetExtractor {
   public:
    CheckerboardExtractor(cv::Size const& patern_size) : TargetExtractor(patern_size) {}

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override {
        return CheckerboardExtractorExtractPixelFeatures(image, pattern_size_);
    }
};

class CircleGridExtractor : public TargetExtractor {
   public:
    CircleGridExtractor(cv::Size const& patern_size, bool const asymmetric)
        : TargetExtractor(patern_size), asymmetric_{asymmetric} {}

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override {
        return CirclegridExtractorExtractPixelFeatures(image, pattern_size_, asymmetric_);
    }

   private:
    bool asymmetric_;
};

std::unique_ptr<TargetExtractor> CreateTargetExtractor(const TargetType type) {
    cv::Size const pattern_size{4, 3};  // comes from config file in the future

    // TODO(Jack): Add aprilgrid condition!
    if (type == TargetType::Checkerboard) {
        return std::make_unique<CheckerboardExtractor>(pattern_size);
    } else {
        bool const asymmetric{false};  // comes from config file in the future
        return std::make_unique<CircleGridExtractor>(pattern_size, asymmetric);
    }
}

}  // namespace reprojection_calibration::feature_extraction

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestBuildExtractorCheckerboard) {
    cv::Size const pattern_size{4, 3};
    int const square_size_pixels{50};
    cv::Mat const checkerboard{GenerateCheckerboard(pattern_size, square_size_pixels)};

    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::Checkerboard)};

    auto const pixels{extractor->Extract(checkerboard)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), pattern_size.height * pattern_size.width);
}

TEST(FeatureExtraction, TestBuildExtractorCircleGrid) {
    cv::Size const pattern_size{4, 3};  // Must match size in CreateTargetExtractor!!! Will be fixed soon.
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};
    bool const asymmetric{false};
    cv::Mat const circle_grid{
        GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::CircleGrid)};
    auto const pixels{extractor->Extract(circle_grid)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), pattern_size.height * pattern_size.width);
}
