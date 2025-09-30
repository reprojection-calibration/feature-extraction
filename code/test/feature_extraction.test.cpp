
#include <gtest/gtest.h>

#include "feature_extraction/target_extraction.hpp"
#include "target_extractors.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestBuildExtractorCheckerboard) {
    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::Checkerboard)};

    EXPECT_TRUE(dynamic_cast<CheckerboardExtractor*>(extractor.get()));
}

TEST(FeatureExtraction, TestBuildExtractorCircleGrid) {
    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::CircleGrid)};

    EXPECT_TRUE(dynamic_cast<CircleGridExtractor*>(extractor.get()));
}
