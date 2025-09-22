#include "checkerboard_extractor.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Should return optional based on "pattern_found"
Eigen::MatrixX2d CheckerboardExtractorExtractPixelFeatures(cv::Mat const& image, cv::Size const pattern_size) {
    std::vector<cv::Point2f> corners;
    bool const pattern_found{cv::findChessboardCorners(
        image, pattern_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)};

    if (pattern_found) {
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }

    // TODO(Jack): Do we need this conversion function in some central location?
    Eigen::MatrixX2d corners_matrix(std::size(corners), 2);
    for (Eigen::Index i = 0; i < corners_matrix.rows(); i++) {
        corners_matrix.row(i)[0] = corners[i].x;
        corners_matrix.row(i)[1] = corners[i].y;
    }

    // TODO(Jack): Figure out what order these come in so we can align them with the 3D geometry
    return corners_matrix;
}

// NOTE(Jack): I think we can get the feature location simply by making a grid index array with the internal row/col
// size multiplied by the dimensions.
Eigen::MatrixX3d CheckerboardExtractorExtractPointFeatures(cv::Size const pattern_size,
                                                           double const unit_dimension_meters_) {
    // TODO(Jack): There has to be a much more eloquent and clear way to create this with eigen linear space operations
    Eigen::MatrixX3d corner_locations(pattern_size.height * pattern_size.width, 3);

    double const z{0};
    for (int row{0}; row < pattern_size.height; row++) {
        for (int col{0}; col < pattern_size.width; col++) {
            double const x{row * unit_dimension_meters_};
            double const y{col * unit_dimension_meters_};
            corner_locations.row((row * pattern_size.width) + col) = Eigen::Vector3d{x, y, z};
        }
    }

    return corner_locations;
}

// TODO(Jack): Should return optional based on "pattern_found"
Eigen::MatrixX2d CirclegridExtractorExtractPixelFeatures(cv::Mat const& image, cv::Size const pattern_size,
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
    cv::drawChessboardCorners(image, pattern_size, cv::Mat(corners), pattern_found);
    static_cast<void>(pattern_found);  // REMOVE

    // TODO(Jack): Do we need this conversion function in some central location?
    Eigen::MatrixX2d corners_matrix(std::size(corners), 2);
    for (Eigen::Index i = 0; i < corners_matrix.rows(); i++) {
        corners_matrix.row(i)[0] = corners[i].x;
        corners_matrix.row(i)[1] = corners[i].y;
    }

    // TODO(Jack): Figure out what order these come in so we can align them with the 3D geometry
    return corners_matrix;
}

}  // namespace reprojection_calibration::feature_extraction