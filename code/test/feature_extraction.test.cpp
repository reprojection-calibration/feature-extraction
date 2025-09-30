
#include <gtest/gtest.h>

#include "feature_extraction/target_extraction.hpp"
#include "target_extractors.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(FeatureExtraction, TestCreateTargetExtractorCheckerboard) {
    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::Checkerboard)};

    EXPECT_TRUE(dynamic_cast<CheckerboardExtractor*>(extractor.get()));
}

TEST(FeatureExtraction, TestCreateTargetExtractorCircleGrid) {
    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::CircleGrid)};

    EXPECT_TRUE(dynamic_cast<CircleGridExtractor*>(extractor.get()));
}

TEST(FeatureExtraction, TestCreateTargetExtractorAprilGrid3) {
    std::unique_ptr<TargetExtractor> const extractor{CreateTargetExtractor(TargetType::AprilGrid3)};

    EXPECT_TRUE(dynamic_cast<AprilGrid3Extractor*>(extractor.get()));
}
