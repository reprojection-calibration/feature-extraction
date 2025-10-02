#include "target_extractors.hpp"

#include <iostream>

extern "C" {
#include "generated_apriltag_code/tagCustom36h11.h"
}

#include "eigen_utilities.hpp"

namespace reprojection_calibration::feature_extraction {

CheckerboardExtractor::CheckerboardExtractor(cv::Size const& pattern_size)
    : TargetExtractor(pattern_size), point_indices_{GenerateGridIndices(pattern_size_.height, pattern_size_.width)} {}

std::optional<FeatureFrame> CheckerboardExtractor::Extract(cv::Mat const& image) const {
    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findChessboardCorners(
        image, pattern_size_, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)};

    if (not pattern_found) {
        return std::nullopt;
    }

    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    return FeatureFrame{ToEigen(corners), point_indices_};
}

CircleGridExtractor::CircleGridExtractor(cv::Size const& pattern_size, bool const asymmetric)
    : TargetExtractor(pattern_size), asymmetric_{asymmetric} {
    if (asymmetric_) {
        // NOTE(Jack): In consideration of how the circle grid extractor works given the current coordinate conventions
        // (and possible just due to how opencv works) we reverse the order of the width and height here for the
        // asymmetric case!
        Eigen::ArrayX2i grid{GenerateGridIndices(pattern_size.width, pattern_size.height)};

        // NOTE(Jack): Eigen does not provide direct way to apply the modulo operator, so we follow a method using a
        // unaryExpr() that we adopted from here
        // (https://stackoverflow.com/questions/35798698/eigen-matrix-library-coefficient-wise-modulo-operation)
        // TODO(Jack): Combine this with the logic for the asymmetric target generation!
        // TODO(Jack): Simplify this extremely by understanding we always just need to start at zero and increment by
        // two to create these data structures we want!
        Eigen::ArrayXi const is_even{
            ((grid.rowwise().sum().unaryExpr([](int const x) { return x % 2; })) == 0).cast<int>()};
        Eigen::ArrayXi const mask{MaskIndices(is_even)};

        // WARN(Jack): I think these work but they are not the row major "top-right" coordinate frame that we normally
        // image when we are working on images. That being said I believe the IDs are at least consistent!
        point_indices_ = grid(mask, Eigen::all);
    } else {
        point_indices_ = GenerateGridIndices(pattern_size_.height, pattern_size_.width);
    }
}

std::optional<FeatureFrame> CircleGridExtractor::Extract(cv::Mat const& image) const {
    // cv::CALIB_CB_CLUSTERING - "uses a special algorithm for grid detection. It is more robust to perspective
    // distortions but much more sensitive to background clutter." - if I do not use this then I think I need to do
    // some tuning about what acceptable sizes and spacing are for the circle grid. For now this will do.
    // TODO(Jack): This is not so clean here because we will have to repeat all options (ex.
    // cv::CALIB_CB_CLUSTERING) even though those will probably be the same for both cases. Keep your eyes peeled
    // for associated problems!
    int const extraction_options{asymmetric_ ? cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_ASYMMETRIC_GRID
                                             : cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_SYMMETRIC_GRID};

    // NOTE(Jack): Something which violates the principle of least surprise is how OpenCV deals with the dimension of
    // asymmetric circle grids. There are two things which are curious to me; #1 that we have to switch the height and
    // width order for the asymmetric case and #2 that we need to divide one of the dimension by two! But this is what
    // works for now.
    // TODO(Jack): Confirm the dimensions in the target generation logic are consistent and correct!
    cv::Size pattern_size{pattern_size_};
    if (asymmetric_) {
        pattern_size = cv::Size{pattern_size_.height / 2, pattern_size_.width};
    }

    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findCirclesGrid(image, pattern_size, corners, extraction_options)};

    if (not pattern_found) {
        return std::nullopt;
    }

    return FeatureFrame{ToEigen(corners), point_indices_};
}

// TODO(Jack): Are we using pattern size here? Or is this just here for fun?
// WARN(Jack): Use of the tagCustom36h11 and all settings are hardcoded here! This means no on can select another
// family. Find a way to make this configurable if possible, but it will likely require recompilation.
AprilGrid3Extractor::AprilGrid3Extractor(cv::Size const& pattern_size)
    : TargetExtractor(pattern_size),
      tag_family_{AprilTagFamily{tagCustom36h11_create(), tagCustom36h11_destroy}},
      tag_detector_{AprilTagDetector{tag_family_, {2.0, 0.0, 1, false, false}}} {}

std::optional<FeatureFrame> AprilGrid3Extractor::Extract(cv::Mat const& image) const {
    std::vector<AprilTagDetection> const raw_detections{tag_detector_.Detect(image)};
    if (std::size(raw_detections) == 0) {
        return std::nullopt;
    }

    Eigen::MatrixX2d corners{4 * std::size(raw_detections), 2};
    for (size_t i{0}; i < std::size(raw_detections); ++i) {
        // ERROR(Jack): The homography can launch the corners outside the bound of the image, this is currently not
        // handled, and how that shows up in our code is not yet clear (2.10.2025).
        Eigen::Matrix<double, 4, 2> const extraction_corners{
            EstimateExtractionCorners(raw_detections[i].H, std::sqrt(tag_family_.tag_family->nbits))};
        Eigen::Matrix<double, 4, 2> const refined_extraction_corners{RefineCorners(image, extraction_corners)};

        corners.block<4, 2>(4 * i, 0) = refined_extraction_corners;
    }

    // TODO(Jack): Make corner and point naming consistent!
    return FeatureFrame{corners, CornerIndices(pattern_size_, raw_detections)};
}

// TODO(Jack): This is not a very eloquent implementation, but its gets the job done for now and the tests pass!
Eigen::ArrayX2i AprilGrid3Extractor::CornerIndices(cv::Size const& pattern_size,
                                                   std::vector<AprilTagDetection> const& detections) {
    std::vector<int> mask_vec;
    for (auto const& detection : detections) {
        int const i{static_cast<int>(detection.id / pattern_size.width)};
        int const j{detection.id % pattern_size.width};

        // TODO(Jack): Align my imagination of the order and indices of the corners with the april tag implementation
        int const corner_0{(2 * (2 * i) * pattern_size.width) + (2 * j)};
        int const corner_1{corner_0 + 1};
        int const corner_2{corner_0 + (2 * pattern_size.width)};
        int const corner_3{corner_2 + 1};

        mask_vec.push_back(corner_0);
        mask_vec.push_back(corner_1);
        mask_vec.push_back(corner_2);
        mask_vec.push_back(corner_3);
    }

    Eigen::ArrayX2i const grid{GenerateGridIndices(2 * pattern_size.height, 2 * pattern_size.width)};
    Eigen::ArrayXi const mask{ToEigen(mask_vec)};

    return grid(mask, Eigen::all);
}

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
Eigen::Matrix<double, 4, 2> AprilGrid3Extractor::EstimateExtractionCorners(Eigen::Matrix3d const& H,
                                                                           int const sqrt_num_bits) {
    // NOTE(Jack): These corners have been reordered from how they are listed in the april tag documentation. The
    // current ordering matches our generated targets grid row/column indexing.
    Eigen::Matrix<double, 4, 2> const canonical_corners{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    double const corner_offset_scale{(sqrt_num_bits / 2.0 + 2.0) / (sqrt_num_bits / 2.0 + 1.0)};

    Eigen::Matrix<double, 4, 2> extraction_corners{
        (H * (corner_offset_scale * canonical_corners).rowwise().homogeneous().transpose())
            .transpose()
            .rowwise()
            .hnormalized()};

    return extraction_corners;
}

Eigen::Matrix<double, 4, 2> AprilGrid3Extractor::RefineCorners(cv::Mat const& image,
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