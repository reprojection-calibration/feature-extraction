#include "feature_extraction/target_extraction.hpp"

#include "target_extractors.hpp"

namespace reprojection_calibration::feature_extraction {

std::unique_ptr<TargetExtractor> CreateTargetExtractor(const TargetType type) {
    cv::Size const pattern_size{4, 3};  // comes from config file in the future
    double const unit_dimension{0.5};   // comes from config file in the future

    if (type == TargetType::Checkerboard) {
        return std::make_unique<CheckerboardExtractor>(pattern_size, unit_dimension);
    } else if (type == TargetType::CircleGrid) {
        bool const asymmetric{false};  // comes from config file in the future
        return std::make_unique<CircleGridExtractor>(pattern_size, unit_dimension, asymmetric);
    } else {
        return std::make_unique<AprilGrid3Extractor>(pattern_size, unit_dimension);
    }
}

}  // namespace reprojection_calibration::feature_extraction