#include "feature_extraction/feature_extraction.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Where are we getting the 3d information!?
Eigen::MatrixX2d ExtractCheckerboardFeatures(cv::Mat const& image, cv::Point const& checkerboard_dimension) {
    std::vector<cv::Point2f> corners;

    bool const pattern_found{cv::findChessboardCorners(
        image, checkerboard_dimension, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)};

    if (pattern_found) {
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }
    cv::drawChessboardCorners(image, checkerboard_dimension, cv::Mat(corners), pattern_found);

    // TODO(Jack): Do we need this conversion function in some central location?
    Eigen::MatrixX2d corners_matrix(std::size(corners), 2);
    for (Eigen::Index i = 0; i < corners_matrix.rows(); i++) {
        corners_matrix.row(i)[0] = corners[i].x;
        corners_matrix.row(i)[1] = corners[i].y;
    }

    return corners_matrix;
}

}  // namespace reprojection_calibration::feature_extraction