#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

struct ExtractedTarget {
    Eigen::MatrixX2d pixels;
    Eigen::MatrixX3d points;
    Eigen::ArrayXi indices;
};

}  // namespace reprojection_calibration::feature_extraction