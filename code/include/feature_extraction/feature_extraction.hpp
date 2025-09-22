#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Are we doing to make the same design choice of having the "target" concept permeate the entire repo?
// Maybe that actually makes sense?
struct ExtractedTarget {
    Eigen::MatrixX2d pixels;
    Eigen::MatrixX3d points;
    Eigen::ArrayXi indices;
};

}  // namespace reprojection_calibration::feature_extraction