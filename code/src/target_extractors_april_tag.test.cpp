
#include <gtest/gtest.h>

#include "feature_extraction/april_tag_cpp_wrapper.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include <apriltag/apriltag.h>

#include "feature_extraction/generated_apriltag_code/tagCustom36h11.h"
}

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractorsAprilTag, HHH) {
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetector const tag_detector{tag_family_handler, {2.0, 0.0, 1, false, false}};

    Eigen::MatrixXi const code_matrix{
        CalculateCodeMatrix(tag_family_handler.tag_family->nbits, tag_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(bit_size_pixel, code_matrix)};

    std::vector<AprilTagDetection> const raw_detections{tag_detector.Detect(april_tag)};

    // Extract
    Eigen::Matrix<double, 4, 2> const extraction_corners{EstimateExtractionCorners(raw_detections[0].H)};
    Eigen::Matrix<double, 4, 2> const gt_extraction_corner{
        {19.75, 122.25}, {122.25, 122.25}, {122.25, 19.75}, {19.75, 19.75}};
    EXPECT_TRUE(extraction_corners.isApprox(gt_extraction_corner, 1e-6));

    // Refine
    Eigen::Matrix<double, 4, 2> const refined_extraction_corners{
        RefineExtractionCorners(april_tag, extraction_corners)};
    Eigen::Matrix<double, 4, 2> const gt_refined_extraction_corner{{19.819417953491211, 119.27910614013672},
                                                                   {119.13014984130859, 119.13014984130859},
                                                                   {119.27910614013672, 19.819416046142578},
                                                                   {19.685731887817383, 19.685731887817383}};
    EXPECT_TRUE(refined_extraction_corners.isApprox(gt_refined_extraction_corner, 1e-6));

    // WARN(Jack): Honestly the size of the error between the extracted and refined values is suprising. Two pixels in
    // the worst case! I think the tags are now created properly, is it maybe just a property of the extractor?
}
