#include "target_generators_april_tag.hpp"

#include "grid_utilities.hpp"

namespace reprojection_calibration::feature_extraction {

// What if we want metric information, we work in pixel space now but we might want metric sizes eventually! This should
// be considered for all target types potentially!
// ERROR(Jack): We need to define a type that contains the tag family and its code, plus the number of bits. That
// information is currently hardcoded!
// WARN(Jack): Our aprilboard orientation is completely different than the Kalibr one! In kalibr tag zero is in the
// bottom right corner of the generated pdf.
cv::Mat GenerateAprilBoard(cv::Size const& pattern_size, int const bit_size_pixels, int const num_bits,
                           uint64_t const tag_family[]) {
    int const april_tag_size_pixels{(8 * bit_size_pixels) + (static_cast<int>(std::sqrt(num_bits)) * bit_size_pixels)};

    int const height{pattern_size.height * april_tag_size_pixels};
    int const width{pattern_size.width * april_tag_size_pixels};
    cv::Mat april_board{cv::Mat::zeros(height, width, CV_8UC1)};

    // Place tags
    Eigen::ArrayX2i const tag_grid{GenerateGridIndices(pattern_size.height, pattern_size.width)};
    for (Eigen::Index i{0}; i < tag_grid.rows(); ++i) {
        Eigen::Array2i const indices{tag_grid.row(i)};
        cv::Point const top_left_corner{((april_tag_size_pixels)*indices(1)), ((april_tag_size_pixels)*indices(0))};
        cv::Point const bottom_right_corner{top_left_corner.x + april_tag_size_pixels,
                                            top_left_corner.y + april_tag_size_pixels};
        cv::Rect const roi{cv::Rect(top_left_corner, bottom_right_corner)};

        // TODO(Jack): Can we somehow clean up the indexing logic in here?
        cv::Mat const april_tag_i{
            GenerateAprilTag(num_bits, tag_family[(indices(0) * pattern_size.width) + indices(1)], bit_size_pixels)};

        april_tag_i.copyTo(april_board(roi));
    }

    return april_board;
}

cv::Mat GenerateAprilTag(Eigen::MatrixXi const& code_matrix, int const bit_size_pixels) {
    int const border_thickness_pixels{8 * bit_size_pixels};  // Three mainly white rings and one black ring - times two
    int const data_region_size_bits{static_cast<int>(code_matrix.rows())};  // One side of the square

    int const height{border_thickness_pixels + (data_region_size_bits * bit_size_pixels)};
    int const width{height};  // Tags are square
    cv::Mat april_tag{255 * cv::Mat::ones(height, width, CV_8UC1)};

    // Hacky way to draw the surrounding black edge rectangle - we cannot use cv::rectangle directly because to maintain
    // constant thickness it will round the rectangle corners.
    {
        // Fill in the center black
        cv::Point const top_left_corner{2 * bit_size_pixels, 2 * bit_size_pixels};
        cv::Point const bottom_right_corner{(6 + data_region_size_bits) * bit_size_pixels,
                                            (6 + data_region_size_bits) * bit_size_pixels};
        cv::rectangle(april_tag, top_left_corner, bottom_right_corner, (0), -1);
    }
    {
        // Fill back in the center data part white - leaving a black rim
        cv::Point const top_left_corner{3 * bit_size_pixels, 3 * bit_size_pixels};
        cv::Point const bottom_right_corner{(5 + data_region_size_bits) * bit_size_pixels,
                                            (5 + data_region_size_bits) * bit_size_pixels};
        cv::rectangle(april_tag, top_left_corner, bottom_right_corner, (255), -1);
    }

    {
        // Now that we have the black border put in the corner sharping elements
        cv::Mat corner_element{cv::Mat::zeros(2 * bit_size_pixels, 2 * bit_size_pixels, CV_8UC1)};
        cv::Point const top_left_corner{0, 0};
        // ERROR(Jack): It is possible that we have an "off by one" with all pixel coordinates - I only noticed here
        // because the size ie so small it made it noticeable.
        cv::Point const bottom_right_corner{bit_size_pixels - 1, bit_size_pixels - 1};
        cv::rectangle(corner_element, top_left_corner, bottom_right_corner, (255), -1);

        // Put in the corner element into all four corners of the tag
        cv::Rect const roi{cv::Rect(cv::Point{0, 0}, cv::Point{2 * bit_size_pixels, 2 * bit_size_pixels})};
        corner_element.copyTo(april_tag(roi));
        cv::rotate(april_tag, april_tag, cv::ROTATE_90_CLOCKWISE);
        corner_element.copyTo(april_tag(roi));
        cv::rotate(april_tag, april_tag, cv::ROTATE_90_CLOCKWISE);
        corner_element.copyTo(april_tag(roi));
        cv::rotate(april_tag, april_tag, cv::ROTATE_90_CLOCKWISE);
        corner_element.copyTo(april_tag(roi));
        cv::rotate(april_tag, april_tag, cv::ROTATE_90_CLOCKWISE);  // Back to starting orientation
    }

    // TODO(Jack): This logic is practically exactly the same as in the checkerboard generation... is there a practical
    // way to DRY ourselves here?
    Eigen::ArrayX2i const grid{GenerateGridIndices(data_region_size_bits, data_region_size_bits)};
    for (Eigen::Index i{0}; i < grid.rows(); ++i) {
        Eigen::Array2i const indices{grid.row(i)};

        if (code_matrix(indices(0), indices(1))) {
            cv::Point const top_left_corner{(bit_size_pixels * indices(1)) + (border_thickness_pixels / 2),
                                            (bit_size_pixels * indices(0)) + (border_thickness_pixels / 2)};
            cv::Point const bottom_right_corner{top_left_corner.x + bit_size_pixels,
                                                top_left_corner.y + bit_size_pixels};
            cv::rectangle(april_tag, top_left_corner, bottom_right_corner, (0), -1);
        }
    }

    cv::Point const top_left_corner{2 * bit_size_pixels, 2 * bit_size_pixels};
    cv::Point const bottom_right_corner{(6 + data_region_size_bits) * bit_size_pixels,
                                        (6 + data_region_size_bits) * bit_size_pixels};
    cv::rectangle(april_tag, top_left_corner, bottom_right_corner, (0));

    return april_tag;
}

cv::Mat GenerateAprilTag(int const num_bits, unsigned long long const tag_code, int const bit_size_pixel) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(num_bits, tag_code)};

    return GenerateAprilTag(code_matrix, bit_size_pixel);
}

// Modeled on ImageLayout.java renderToArray()
// TODO(Jack): Consider typedef for unsigned long long type used everywhere
// TODO(Jack): Rewrite this code without requiring the 90 degree rotations! Just make it all at once without rotations.
Eigen::MatrixXi CalculateCodeMatrix(int const num_bits, unsigned long long tag_code) {
    int const sqrt_num_bits{static_cast<int>(std::sqrt(num_bits))};

    Eigen::MatrixXi code_matrix(sqrt_num_bits, sqrt_num_bits);
    for (int k{0}; k < 4; ++k) {
        for (int i{0}; i <= sqrt_num_bits / 2; ++i) {
            for (int j{i}; j < sqrt_num_bits - 1 - i; ++j) {
                unsigned long long bit_sign{(tag_code & (static_cast<unsigned long long>(1) << (num_bits - 1)))};
                code_matrix(j, i) = not bit_sign;  // I SWITCHED I AND J AND IT BASICALLY STARTED WORKING

                tag_code = tag_code << 1;
            }
        }
        code_matrix = Rotate90(code_matrix, true);
    }

    // Set center pixel if there is one (i.e. odd numbers of rows and columns)
    if (sqrt_num_bits % 2 != 0) {
        unsigned long long bit_sign{(tag_code & (static_cast<unsigned long long>(1) << (num_bits - 1)))};
        // TODO(Jack): Static cast
        code_matrix(sqrt_num_bits / 2, sqrt_num_bits / 2) =
            not bit_sign;  // I SWITCHED I AND J AND IT BASICALLY STARTED WORKING
    }
    // WARN(Jack): Why I need this transpose I am not 100% sure, but compared to the official implementation we look
    // mirrored without this step.
    return code_matrix.transpose();
}

Eigen::MatrixXi Rotate90(Eigen::MatrixXi const& matrix, bool const clockwise) {
    Eigen::MatrixXi const matrix_star{matrix.transpose()};

    return clockwise ? matrix_star.rowwise().reverse().eval() : matrix_star.colwise().reverse().eval();
}

}  // namespace reprojection_calibration::feature_extraction