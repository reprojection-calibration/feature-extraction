#pragma once

#include <Eigen/Dense>

namespace reprojection_calibration::feature_extraction {

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols);

}  // namespace reprojection_calibration::feature_extraction