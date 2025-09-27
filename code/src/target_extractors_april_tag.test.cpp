
#include <gtest/gtest.h>

#include "../include/feature_extraction/april_tag_cpp_wrapper.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include <apriltag/apriltag.h>

#include "feature_extraction/generated_apriltag_code/tagCustom36h11.h"
}

// {}
// []

namespace reprojection_calibration::feature_extraction {

// From the apriltag documentation (https://github.com/AprilRobotics/apriltag/blob/master/apriltag.h)
//
//      The 3x3 homography matrix describing the projection from an "ideal" tag (with corners at (-1,1), (1,1), (1,-1),
//      and (-1,-1)) to pixels in the image.
//
// Here the "corner" positions correspond to the four corners on the inside of the black ring that defines the "quad" of
// an April Tag 3. In the tags designed for use in the April Board 3, the corners that we want to extract and use are
// found on the outside of this black ring, at the intersection of the black ring and the corner element. This
// intersection is designed to provide the characteristic checkerboard like intersection which can be refined using the
// cv::cornerSubPix() function to provide nearly exact corner pixel coordinates.
// ADD , int const num_bits
Eigen::Matrix<double, 4, 2> EstimateExtractionCorners(Eigen::Matrix3d const& H) {
    Eigen::Matrix<double, 4, 2> canonical_corners{{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
    canonical_corners *= (4.5 / 3.5);  // USE NUM_BITS

    // REMOVE THE COLWISE HNORMALIZED AND REPLACE WITH ROWWISE
    Eigen::Matrix<double, 4, 2> extraction_corners{
        (H * canonical_corners.rowwise().homogeneous().transpose()).colwise().hnormalized().transpose()};

    return extraction_corners;
}

Eigen::Matrix<double, 4, 2> RefineExtractionCorners(cv::Mat const& image,
                                                    Eigen::Matrix<double, 4, 2> const& extraction_corners) {
    Eigen::Matrix<float, 4, 2> refined_extraction_corners{extraction_corners.cast<float>()};
    cv::Mat cv_view_extraction_corners(refined_extraction_corners.rows(), refined_extraction_corners.cols(), CV_32FC1,
                                       refined_extraction_corners.data());  // cv::cornerSubPix() requires float type

    cv::cornerSubPix(image, cv_view_extraction_corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    return refined_extraction_corners.cast<double>();
}

}  // namespace reprojection_calibration::feature_extraction

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
        {18.28571, 123.71429}, {123.71429, 123.71429}, {123.71429, 18.28571}, {18.28571, 18.28571}};
    EXPECT_TRUE(extraction_corners.isApprox(gt_extraction_corner, 1e-6));

    // Refine
    Eigen::Matrix<double, 4, 2> const refined_extraction_corners{
        RefineExtractionCorners(april_tag, extraction_corners)};
    Eigen::Matrix<double, 4, 2> const gt_refined_extraction_corner{
        {19.47327, 120.10841}, {120.06904, 120.10841}, {120.06904, 19.499969}, {19.47327, 19.499969}};
    EXPECT_TRUE(refined_extraction_corners.isApprox(gt_refined_extraction_corner, 1e-6));

    // REMOVE WHEN WE ARE CONFIDENT THIS WORKS!
    // for (int i{0}; i < 4; ++i) {
    //    cv::circle(april_tag, cv::Point(refined_extraction_corners.row(i)[0], refined_extraction_corners.row(i)[1]),
    //    1,
    //              cv::Scalar(127), 1, cv::LINE_8);
    //}
}
