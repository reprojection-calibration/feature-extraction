#include "target_extractors.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Should return optional based on "pattern_found"
std::optional<Eigen::MatrixX2d> CheckerboardExtractorExtractPixelFeatures(cv::Mat const& image,
                                                                          cv::Size const pattern_size) {
    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findChessboardCorners(
        image, pattern_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)};

    if (not pattern_found) {
        return std::nullopt;
    }

    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    return ToEigen(corners);
}

// TODO(Jack): Should return optional based on "pattern_found"
std::optional<Eigen::MatrixX2d> CirclegridExtractorExtractPixelFeatures(cv::Mat const& image,
                                                                        cv::Size const pattern_size,
                                                                        bool const asymmetric) {
    // cv::CALIB_CB_CLUSTERING - "uses a special algorithm for grid detection. It is more robust to perspective
    // distortions but much more sensitive to background clutter." - if I do not use this then I think I need to do some
    // tuning about what acceptable sizes and spacing are for the circle grid. For now this will do.
    // TODO(Jack): This is not so clean here because we will have to repeat all options (ex. cv::CALIB_CB_CLUSTERING)
    // even though those will probably be the same for both cases. Keep your eyes peeled for associated problems!
    int const extraction_options{asymmetric ? cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_ASYMMETRIC_GRID
                                            : cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_SYMMETRIC_GRID};

    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findCirclesGrid(image, pattern_size, corners, extraction_options)};

    if (not pattern_found) {
        return std::nullopt;
    }

    return ToEigen(corners);
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