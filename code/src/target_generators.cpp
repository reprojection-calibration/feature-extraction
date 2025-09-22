#include "target_generators.hpp"

#include <iostream>

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Consider using cv:Size instead of rows and cols.
cv::Mat GenerateCheckerboard(int const internal_rows, int const internal_cols, int const unit_dimension_pixels) {
    // TODO(Jack): Which concept should the user be familiar with, the "internal rows/cols" or the checkerboards
    // rows/cols themselves
    int const rows{internal_rows + 1};
    int const cols{internal_cols + 1};

    int const buffer{2 * unit_dimension_pixels};  // White buffer space around checkerboard aids extraction
    int const height{(unit_dimension_pixels * rows) + buffer};
    int const width{(unit_dimension_pixels * cols) + buffer};
    cv::Mat checkerboard{255 * cv::Mat::ones(height, width, CV_8UC1)};  // Start with white image

    Eigen::ArrayX2i const grid{GenerateGridIndices(rows, cols)};
    for (Eigen::Index i{0}; i < grid.rows(); ++i) {
        Eigen::Array2i const indices{grid.row(i)};

        // This condition gives us an asymmetric grid - like that of  checkerboard/chessboard
        if (indices.sum() % 2 == 0) {
            cv::Point const top_left_corner{(unit_dimension_pixels * indices(1)) + (buffer / 2),
                                            (unit_dimension_pixels * indices(0)) + (buffer / 2)};
            cv::Point const bottom_right_corner{top_left_corner.x + unit_dimension_pixels,
                                                top_left_corner.y + unit_dimension_pixels};
            cv::rectangle(checkerboard, top_left_corner, bottom_right_corner, (0), -1);
        }
    }

    return checkerboard;
}

// NOTE(Jack): The unit dimension for the circle is its radius!
// NOTE(Jack): The circles themselves are the features, not the intersection between the circles, therefore the indexing
// logic will be different than by the checkerboard - i.e the checkboard always works in the rows+1 or cols+1 space,
// whereas the circle grid will works on rows and cols directly.
cv::Mat GenerateCircleGrid(int rows, int cols, int const unit_dimension_pixels, int const unit_spacing_pixels,
                           bool const asymmetric) {
    // TODO ADD UNIT SPACING - what does this mean?
    int const circle_size{2 * unit_dimension_pixels};
    // circles + spacing + edge buffer
    int const height{(circle_size * rows) + (unit_spacing_pixels * (rows - 3)) + (2 * circle_size)};
    int const width{(circle_size * cols) + (unit_spacing_pixels * (cols - 3)) + (2 * circle_size)};
    cv::Mat circlgrid{255 * cv::Mat::ones(height, width, CV_8UC1)};

    Eigen::ArrayX2i const grid{GenerateGridIndices(rows, cols)};
    for (Eigen::Index i{0}; i < grid.rows(); ++i) {
        Eigen::Array2i const indices{grid.row(i)};

        if (asymmetric and indices.sum() % 2 != 0) {
            continue;
        }

        cv::Point const center{
            unit_dimension_pixels + circle_size + (unit_spacing_pixels * (indices(1) - 1)) + (circle_size * indices(1)),
            unit_dimension_pixels + circle_size + (unit_spacing_pixels * (indices(0) - 1)) +
                (circle_size * indices(0))};
        cv::circle(circlgrid, center, unit_dimension_pixels, (0), -1);
    }

    return circlgrid;
}

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols) {
    Eigen::ArrayXi const row_indices = Eigen::ArrayXi::LinSpaced(rows * cols, 0, rows - 1);
    Eigen::ArrayXi const col_indices = Eigen::ArrayXi::LinSpaced(cols, 0, cols).colwise().replicate(rows);

    Eigen::ArrayX2i grid_indices(rows * cols, 2);
    grid_indices.col(0) = row_indices;
    grid_indices.col(1) = col_indices;

    return grid_indices;
}
}  // namespace reprojection_calibration::feature_extraction