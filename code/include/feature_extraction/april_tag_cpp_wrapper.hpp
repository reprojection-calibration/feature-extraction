#pragma once

extern "C" {
#include <apriltag/apriltag.h>
}

#include <Eigen/Dense>
#include <functional>
#include <opencv2/opencv.hpp>

// This is my attempt to RAII-ify the  C code from the apriltag repository. The main thing I try to fight here is
// manually having to deallocate memory. For the detector and tag detections themselves that is relatively easy because
// they have generic creation and destruction functions. For the tag family it is trickier because the generated code is
// actually specific to each one (see comment in AprilTagFamily).
//
// WARN(Jack): The const correctness and memory safety of all apriltag related code is not clear at this point
// (26.09.2025)! I am 99% sure that there are some big footguns in here, and we will find some "presents" later.

namespace reprojection_calibration::feature_extraction {

struct AprilTagFamily {
    // WARN(Jack): If the user passes mismatched _tag_family and _tag_family_destroy this class will not do what they
    // actually want it to! If for example the user did the following:
    //
    //      AprilTagFamily(tagCustom36h11_create(), tag25h9_destroy)
    //
    // Then this is a huge footgun because the created tag family does not match the one that will be destroyed when the
    // destructor is called. Because of the nature of the generated code there is no way to enforce that matching pairs
    // are passed, or that the proper destructor is called automatically. This is our best attempt given our options.
    AprilTagFamily(apriltag_family_t* _tag_family, std::function<void(apriltag_family_t*)> _tag_family_destroy)
        : tag_family{_tag_family}, tag_family_destroy{std::move(_tag_family_destroy)} {}

    ~AprilTagFamily() { tag_family_destroy(tag_family); }

    apriltag_family_t* tag_family;

   private:
    std::function<void(apriltag_family_t*)> tag_family_destroy;
};

struct AprilTagDetection {
    AprilTagDetection(apriltag_detection_t const& raw_detection) : id{raw_detection.id} {
        // Grab the homography
        using RowMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        Eigen::Map<RowMatrix3d> const H_map{raw_detection.H->data};
        H = H_map;

        // Grab the center
        c = Eigen::Vector2d{raw_detection.c[0], raw_detection.c[1]};

        // Grab the points
        for (int i{0}; i < 4; i++) {
            p.row(i) = Eigen::Vector2d{raw_detection.p[i][0], raw_detection.p[i][1]}.transpose();
        }
    }

    int id;
    Eigen::Matrix3d H;
    Eigen::Vector2d c;
    Eigen::Matrix<double, 4, 2> p;
};

struct AprilTagDetector {
    struct AprilTagDetectorSettings {
        double decimate;
        double blur;
        int threads;
        bool debug;
        bool refine_edges;
    };

    AprilTagDetector(AprilTagFamily const& tag_family, AprilTagDetectorSettings const& settings)
        : tag_detector{apriltag_detector_create()} {
        apriltag_detector_add_family(tag_detector, tag_family.tag_family);

        tag_detector->quad_decimate = settings.decimate;
        tag_detector->quad_sigma = settings.blur;
        tag_detector->nthreads = settings.threads;
        tag_detector->debug = settings.debug;
        tag_detector->refine_edges = settings.refine_edges;
    }

    // WARN(Jack): Must be grayscale image
    std::vector<AprilTagDetection> Detect(cv::Mat const& gray) const {
        image_u8_t raw_gray{gray.cols, gray.rows, gray.cols, gray.data};
        zarray_t* raw_detections{apriltag_detector_detect(tag_detector, &raw_gray)};

        std::vector<AprilTagDetection> detections;
        for (int i = 0; i < raw_detections->size; i++) {
            apriltag_detection_t* raw_detection;
            zarray_get(raw_detections, i, &raw_detection);
            detections.emplace_back(*raw_detection);
        }

        apriltag_detections_destroy(raw_detections);

        return detections;
    }

    ~AprilTagDetector() { apriltag_detector_destroy(tag_detector); }

   private:
    apriltag_detector_t* tag_detector;
};

}  // namespace reprojection_calibration::feature_extraction

// TEMPORARY FOR WEBCAM DEMO _ MOVE TO PROPER LOCAION_
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
    canonical_corners *= (5.0 / 4.0);  // USE NUM_BITS

    // REMOVE THE COLWISE HNORMALIZED AND REPLACE WITH ROWWISE
    Eigen::Matrix<double, 4, 2> extraction_corners{
        (H * canonical_corners.rowwise().homogeneous().transpose()).colwise().hnormalized().transpose()};

    return extraction_corners;
}

Eigen::Matrix<double, 4, 2> RefineExtractionCorners(cv::Mat const& image,
                                                    Eigen::Matrix<double, 4, 2> const& extraction_corners) {
    // NOTE(Jack): Eigen is column major by default, but opencv is row major (like the rest of the world...) so we need
    // to specifically specify Eigen::RowMajor here in order for the cv::Mat view to make sense.
    Eigen::Matrix<float, 4, 2, Eigen::RowMajor> refined_extraction_corners{extraction_corners.cast<float>()};
    cv::Mat cv_view_extraction_corners(refined_extraction_corners.rows(), refined_extraction_corners.cols(), CV_32FC1,
                                       refined_extraction_corners.data());  // cv::cornerSubPix() requires float type

    cv::cornerSubPix(image, cv_view_extraction_corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    return refined_extraction_corners.cast<double>();
}

}  // namespace reprojection_calibration::feature_extraction