
#include <gtest/gtest.h>

#include "april_tag_cpp_wrapper.hpp"
#include "target_generators_april_tag.hpp"

extern "C" {
#include <apriltag/apriltag.h>

#include "generated_apriltag_code/tagCustom36h11.h"
}

// {}
// []

namespace reprojection_calibration::feature_extraction {

struct AprilTagDetection {
    AprilTagDetection(apriltag_detection_t const& raw_detection) {
        // Grab the homography
        using RowMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        Eigen::Map<RowMatrix3d> const H_map{raw_detection.H->data};
        H = H_map;

        // Grab the points
        for (int i{0}; i < 4; i++) {
            p.row(i) = Eigen::Vector2d{raw_detection.p[i][0], raw_detection.p[i][1]}.transpose();
        }
    }

    Eigen::Matrix3d H;
    Eigen::Matrix<double, 4, 2> p;
};

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
    canonical_corners *= (4.5 / 3.5); // USE NUM_BITS

    // REMOVE THE COLWISE HNORMALIZED AND REPLACE WITH ROWWISE
    Eigen::Matrix<double, 4, 2> extraction_corners{
        (H * canonical_corners.rowwise().homogeneous().transpose()).colwise().hnormalized().transpose()};

    return extraction_corners;
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

    AprilTagDetections const raw_detections{tag_detector.Detect(april_tag)};
    AprilTagDetection const detection{raw_detections[0]};

    // Draw homography transformed points
    Eigen::Matrix<double, 4, 2> extraction_corners{EstimateExtractionCorners(detection.H)};

    // Do subpixel refinement
    std::vector<cv::Point2f> cv_corners;
    for (Eigen::Index i{0}; i < 4; ++i) {
        Eigen::Vector2d const p{extraction_corners.row(i)};
        cv_corners.push_back(cv::Point2f{static_cast<float>(p(0)), static_cast<float>(p(1))});
    }

    cv::cornerSubPix(april_tag, cv_corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    for (int i{0}; i < 4; ++i) {
        cv::circle(april_tag, cv_corners[i], 1, cv::Scalar(127), 1, cv::LINE_8);
    }

    EXPECT_EQ(1, 2);
}
