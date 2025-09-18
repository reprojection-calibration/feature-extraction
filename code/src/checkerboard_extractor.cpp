#include "feature_extraction/target_extractors.hpp"

namespace reprojection_calibration::feature_extraction {

CheckerboardExtractor::CheckerboardExtractor(cv::Size const& pattern_size, double const unit_dimension_meters)
    : pattern_size_{pattern_size}, unit_dimension_meters_{unit_dimension_meters} {}

std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> CheckerboardExtractor::ExtractTarget(cv::Mat const& image) const {
    return {ExtractPixelFeatures(image), ExtractPointFeatures()};
}

Eigen::MatrixX2d CheckerboardExtractor::ExtractPixelFeatures(cv::Mat const& image) const {
    std::vector<cv::Point2f> corners;

    bool const pattern_found{cv::findChessboardCorners(
        image, pattern_size_, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)};

    if (pattern_found) {
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }
    cv::drawChessboardCorners(image, pattern_size_, cv::Mat(corners), pattern_found);

    // TODO(Jack): Do we need this conversion function in some central location?
    Eigen::MatrixX2d corners_matrix(std::size(corners), 2);
    for (Eigen::Index i = 0; i < corners_matrix.rows(); i++) {
        corners_matrix.row(i)[0] = corners[i].x;
        corners_matrix.row(i)[1] = corners[i].y;
    }

    // TODO(Jack): Figure out what order these come in so we can align them with the 3D geometry
    return corners_matrix;
}

Eigen::MatrixX3d CheckerboardExtractor::ExtractPointFeatures() const {
    // RETURN REAL VALUES
    return Eigen::Matrix3d::Identity();
}

}  // namespace reprojection_calibration::feature_extraction