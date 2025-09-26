#include "target_generators.hpp"

#include "grid_utilities.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Consider using cv:Size instead of rows and cols.
cv::Mat GenerateCheckerboard(cv::Size const& pattern_size, int const square_size_pixels) {
    // TODO(Jack): Which concept should the user be familiar with, the "internal rows/cols" or the checkerboards
    // rows/cols themselves
    int const rows{pattern_size.height + 1};
    int const cols{pattern_size.width + 1};

    int const white_space_border{2 * square_size_pixels};  // White space around checkerboard aids extraction
    int const height{(square_size_pixels * rows) + white_space_border};
    int const width{(square_size_pixels * cols) + white_space_border};
    cv::Mat checkerboard{255 * cv::Mat::ones(height, width, CV_8UC1)};  // Start with white image

    Eigen::ArrayX2i const grid{GenerateGridIndices(rows, cols)};
    for (Eigen::Index i{0}; i < grid.rows(); ++i) {
        Eigen::Array2i const indices{grid.row(i)};

        // This condition gives us an asymmetric grid - like that of  checkerboard/chessboard
        if (indices.sum() % 2 == 0) {
            cv::Point const top_left_corner{(square_size_pixels * indices(1)) + (white_space_border / 2),
                                            (square_size_pixels * indices(0)) + (white_space_border / 2)};
            cv::Point const bottom_right_corner{top_left_corner.x + square_size_pixels,
                                                top_left_corner.y + square_size_pixels};
            cv::rectangle(checkerboard, top_left_corner, bottom_right_corner, (0), -1);
        }
    }

    return checkerboard;
}

// NOTE(Jack): The unit dimension for the circle is its radius!
// NOTE(Jack): The circles themselves are the features, not the intersection between the circles, therefore the indexing
// logic will be different than by the checkerboard - i.e the checkboard always works in the rows+1 or cols+1 space,
// whereas the circle grid will works on rows and cols directly.
cv::Mat GenerateCircleGrid(cv::Size const& pattern_size, int const circle_radius_pixels,
                           int const circle_spacing_pixels, bool const asymmetric) {
    int const circumference{2 * circle_radius_pixels};
    int const white_space_border{2 * circumference};  // White space around checkerboard aids extraction
    // WARN(Jack): Is this "pattern_size.* -3" logic make sense? Are we at risk of a throwing a negative number here? We
    // might need to add some conditions to prevent that.
    int const height{(circumference * pattern_size.height) + (circle_spacing_pixels * (pattern_size.height - 3)) +
                     white_space_border};
    int const width{(circumference * pattern_size.width) + (circle_spacing_pixels * (pattern_size.width - 3)) +
                    white_space_border};
    cv::Mat circlgrid{255 * cv::Mat::ones(height, width, CV_8UC1)};

    Eigen::ArrayX2i const grid{GenerateGridIndices(pattern_size.height, pattern_size.width)};
    for (Eigen::Index i{0}; i < grid.rows(); ++i) {
        Eigen::Array2i const indices{grid.row(i)};

        if (asymmetric and indices.sum() % 2 != 0) {
            continue;
        }

        // THERE IS SOMETHING WRONG! THE WHITE SPACE BORDER IS NOT ONE FULL CIRCLE CIRCUMFERENCE
        cv::Point const center{circle_radius_pixels + (white_space_border / 2) +
                                   (circle_spacing_pixels * (indices(1) - 1)) + (circumference * indices(1)),
                               circle_radius_pixels + (white_space_border / 2) +
                                   (circle_spacing_pixels * (indices(0) - 1)) + (circumference * indices(0))};
        cv::circle(circlgrid, center, circle_radius_pixels, (0), -1);
    }

    return circlgrid;
}

}  // namespace reprojection_calibration::feature_extraction