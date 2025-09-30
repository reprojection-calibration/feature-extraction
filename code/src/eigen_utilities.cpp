#include "eigen_utilities.hpp"

namespace reprojection_calibration::feature_extraction {

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols) {
    Eigen::ArrayXi const row_indices = Eigen::ArrayXi::LinSpaced(rows * cols, 0, rows - 1);
    Eigen::ArrayXi const col_indices = Eigen::ArrayXi::LinSpaced(cols, 0, cols).colwise().replicate(rows);

    Eigen::ArrayX2i grid_indices(rows * cols, 2);
    grid_indices.col(0) = row_indices;
    grid_indices.col(1) = col_indices;

    return grid_indices;
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