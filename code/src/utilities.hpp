#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace reprojection::feature_extraction {

cv::Mat ToGray(cv::Mat const& image);

Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points);

double AlternatingSum(int n, double increment_1, double increment_2);

// Requesting even_only=true has the effect of allowing us to produce asymmetric grids like those required by the
// asymmetric circle grid target.
Eigen::ArrayX2i GenerateGridIndices(int rows, int cols, bool even_only = false);

Eigen::ArrayXi MaskToRowId(Eigen::ArrayX<bool> const& mask);

Eigen::ArrayXi ToEigen(std::vector<int> const& vector);

// Adopted from
// https://stackoverflow.com/questions/62696455/is-there-a-way-to-find-the-median-value-of-coefficients-of-an-eigen-matrix
double Median(Eigen::ArrayXd data);

}  // namespace reprojection::feature_extraction