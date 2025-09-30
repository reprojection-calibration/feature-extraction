#include "target_extractors.hpp"

extern "C" {

#include "feature_extraction/generated_apriltag_code/tagCustom36h11.h"
}

namespace reprojection_calibration::feature_extraction {

CheckerboardExtractor::CheckerboardExtractor(cv::Size const& pattern_size) : TargetExtractor(pattern_size) {}

std::optional<Eigen::MatrixX2d> CheckerboardExtractor::Extract(cv::Mat const& image) const {
    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findChessboardCorners(
        image, pattern_size_, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)};

    if (not pattern_found) {
        return std::nullopt;
    }

    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    return ToEigen(corners);
}

CircleGridExtractor::CircleGridExtractor(cv::Size const& pattern_size, bool const asymmetric)
    : TargetExtractor(pattern_size), asymmetric_{asymmetric} {}

std::optional<Eigen::MatrixX2d> CircleGridExtractor::Extract(cv::Mat const& image) const {
    // cv::CALIB_CB_CLUSTERING - "uses a special algorithm for grid detection. It is more robust to perspective
    // distortions but much more sensitive to background clutter." - if I do not use this then I think I need to do
    // some tuning about what acceptable sizes and spacing are for the circle grid. For now this will do.
    // TODO(Jack): This is not so clean here because we will have to repeat all options (ex.
    // cv::CALIB_CB_CLUSTERING) even though those will probably be the same for both cases. Keep your eyes peeled
    // for associated problems!
    int const extraction_options{asymmetric_ ? cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_ASYMMETRIC_GRID
                                             : cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_SYMMETRIC_GRID};

    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findCirclesGrid(image, pattern_size_, corners, extraction_options)};

    if (not pattern_found) {
        return std::nullopt;
    }

    return ToEigen(corners);
}

// TODO(Jack): Are we using pattern size here? Or is this just here for fun?
// WARN(Jack): Use of the tagCustom36h11 and all settings are hardcoded here! This means no on can select another
// family. Find a way to make this configurable if possible, but it will likely require recompilation.
AprilGrid3Extractor::AprilGrid3Extractor(cv::Size const& pattern_size)
    : TargetExtractor(pattern_size),
      tag_family_{AprilTagFamily{tagCustom36h11_create(), tagCustom36h11_destroy}},
      tag_detector_{AprilTagDetector{tag_family_, {2.0, 0.0, 1, false, false}}} {}

std::optional<Eigen::MatrixX2d> AprilGrid3Extractor::Extract(cv::Mat const& image) const {
    std::vector<AprilTagDetection> const raw_detections{tag_detector_.Detect(image)};
    if (std::size(raw_detections) == 0) {
        return std::nullopt;
    }

    Eigen::MatrixX2d points{4 * std::size(raw_detections), 2};
    for (size_t i{0}; i < std::size(raw_detections); ++i) {
        Eigen::Matrix<double, 4, 2> const extraction_corners{
            EstimateExtractionCorners(raw_detections[i].H, std::sqrt(tag_family_.tag_family->nbits))};
        Eigen::Matrix<double, 4, 2> const refined_extraction_corners{
            RefineExtractionCorners(image, extraction_corners)};

        points.block<4, 2>(4 * i, 0) = refined_extraction_corners;
    }

    return points;
}

std::unique_ptr<TargetExtractor> CreateTargetExtractor(const TargetType type) {
    cv::Size const pattern_size{4, 3};  // comes from config file in the future

    // TODO(Jack): Add aprilgrid condition!
    if (type == TargetType::Checkerboard) {
        return std::make_unique<CheckerboardExtractor>(pattern_size);
    } else if (type == TargetType::CircleGrid) {
        bool const asymmetric{false};  // comes from config file in the future
        return std::make_unique<CircleGridExtractor>(pattern_size, asymmetric);
    } else {
        // WARN(Jack): Pattern size might not be used
        return std::make_unique<AprilGrid3Extractor>(pattern_size);
    }
}

Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points) {
    Eigen::MatrixX2d eigen_points(std::size(points), 2);
    for (Eigen::Index i = 0; i < eigen_points.rows(); i++) {
        eigen_points.row(i)[0] = points[i].x;
        eigen_points.row(i)[1] = points[i].y;
    }

    return eigen_points;
}

}  // namespace reprojection_calibration::feature_extraction