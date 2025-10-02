#include "eigen_utilities.hpp"

#include <numeric>

namespace reprojection_calibration::feature_extraction {

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols) {
    Eigen::ArrayXi const row_indices{Eigen::ArrayXi::LinSpaced(rows * cols, 0, rows - 1)};
    Eigen::ArrayXi const col_indices{Eigen::ArrayXi::LinSpaced(cols, 0, cols).colwise().replicate(rows)};

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

// There has to be a more eloquent way to do this... but it gets the job done :)
Eigen::ArrayXi MaskIndices(Eigen::ArrayXi const& array) {
    std::vector<int> mask_vec;
    mask_vec.reserve(array.rows());

    for (Eigen::Index i{0}; i < array.rows(); i++) {
        if (array(i) == 1) {
            mask_vec.push_back(i);
        }
    }

    Eigen::ArrayXi mask(std::size(mask_vec));
    mask = Eigen::Map<Eigen::ArrayXi>(mask_vec.data(), std::size(mask_vec));

    return mask;
}

}  // namespace reprojection_calibration::feature_extraction