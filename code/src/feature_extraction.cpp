#include "feature_extraction/feature_extraction.hpp"

namespace reprojection_calibration::feature_extraction {

std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> ExtractFeatures(cv::Mat const& image, TargetType const& target_type) {
    (void)image;
    (void)target_type;

    // RETURN REAL VALUES
    return {Eigen::MatrixX2d{}, Eigen::MatrixX3d{}};
}

}  // namespace reprojection_calibration::feature_extraction