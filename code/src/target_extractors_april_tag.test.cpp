
#include <gtest/gtest.h>

#include "target_extractors.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include "feature_extraction/generated_apriltag_code/tagCustom36h11.h"
}

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractorsAprilTag, HHH) {
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    Eigen::MatrixXi const code_matrix{
        CalculateCodeMatrix(tag_family_handler.tag_family->nbits, tag_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(bit_size_pixel, code_matrix)};

    cv::Size const pattern_size{7, 6};  // WARN(Jack): Not actually needed here
    auto const extractor{AprilGrid3Extractor{pattern_size}};

    std::optional<Eigen::MatrixX2d> const pixels{extractor.Extract(april_tag)};

    EXPECT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->rows(), 4);  // One tag

    Eigen::Matrix<double, 4, 2> const gt_pixels{{19.819417953491211, 119.27910614013672},
                                                {119.13014984130859, 119.13014984130859},
                                                {119.27910614013672, 19.819416046142578},
                                                {19.685731887817383, 19.685731887817383}};
    EXPECT_TRUE(pixels.value().isApprox(gt_pixels, 1e-6));
}
