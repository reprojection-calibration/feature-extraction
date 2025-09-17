#include "feature_extraction/feature_extraction.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Where are we getting the 3d information!?
std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> ExtractCheckerboardFeatures(cv::Mat const& image,
                                                                           cv::Point const& checkerboard_dimension) {
    (void)image;
    (void)checkerboard_dimension;

    // RETURN REAL VALUES
    return {Eigen::MatrixX2d{}, Eigen::MatrixX3d{}};
}

}  // namespace reprojection_calibration::feature_extraction