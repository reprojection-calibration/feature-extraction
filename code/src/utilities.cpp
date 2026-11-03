#include "utilities.hpp"

namespace reprojection::feature_extraction {

cv::Mat ToGray(cv::Mat const& img) {
    cv::Mat img_8u;
    if (img.depth() != CV_8U) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        img.convertTo(img_8u, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        img_8u = img;
    }

    cv::Mat img_gray;
    if (img_8u.channels() == 3) {
        cv::cvtColor(img_8u, img_gray, cv::COLOR_BGR2GRAY);
    } else if (img_8u.channels() == 4) {
        cv::cvtColor(img_8u, img_gray, cv::COLOR_BGRA2GRAY);
    } else if (img_8u.channels() == 1) {
        img_gray = img_8u;
    } else {
        throw std::runtime_error("LIBRARY IMPLEMENTATION ERROR: Unsupported number of channels: " +  // LCOV_EXCL_LINE
                                 std::to_string(img_8u.channels()));                                 // LCOV_EXCL_LINE
    }

    return img_gray;
}

Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points) {
    Eigen::MatrixX2d eigen_points(std::size(points), 2);
    for (Eigen::Index i = 0; i < eigen_points.rows(); i++) {
        eigen_points.row(i)[0] = points[i].x;
        eigen_points.row(i)[1] = points[i].y;
    }

    return eigen_points;
}  // LCOV_EXCL_LINE

double AlternatingSum(int const n, double const increment_1, double const increment_2) {
    double sum{0};
    for (int i{0}; i < n; ++i) {
        sum += (i % 2 == 0) ? increment_1 : increment_2;
    }

    return sum;
}

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols, bool const even_only) {
    Eigen::ArrayXi const row_indices{Eigen::ArrayXi::LinSpaced(rows * cols, 0, rows - 1)};
    Eigen::ArrayXi const col_indices{Eigen::ArrayXi::LinSpaced(cols, 0, cols).colwise().replicate(rows)};

    Eigen::ArrayX2i grid_indices(rows * cols, 2);
    grid_indices.col(0) = row_indices;
    grid_indices.col(1) = col_indices;

    if (even_only) {
        // NOTE(Jack): Eigen does not provide direct way to apply the modulo operator, so we follow a method using a
        // unaryExpr() that we adopted from here
        // (https://stackoverflow.com/questions/35798698/eigen-matrix-library-coefficient-wise-modulo-operation)
        Eigen::ArrayX<bool> const is_even_mask{
            ((grid_indices.rowwise().sum().unaryExpr([](int const x) { return x % 2; })) == 0)};
        Eigen::ArrayXi const even_row_ids{MaskToRowId(is_even_mask)};

        return grid_indices(even_row_ids, Eigen::all);
    }

    return grid_indices;
}

// There has to be a more eloquent way to do this... but it gets the job done :)
Eigen::ArrayXi MaskToRowId(Eigen::ArrayX<bool> const& mask) {
    std::vector<int> row_ids;
    row_ids.reserve(mask.rows());

    for (Eigen::Index i{0}; i < mask.rows(); i++) {
        if (mask(i) == true) {
            row_ids.push_back(i);
        }
    }

    return ToEigen(row_ids);
}

Eigen::ArrayXi ToEigen(std::vector<int> const& vector) {
    return Eigen::Map<Eigen::ArrayXi const>(vector.data(), std::size(vector));
}

double Median(Eigen::ArrayXd data) {
    std::sort(data.begin(), data.end());

    return data.size() % 2 == 0 ? data.segment((data.size() - 2) / 2, 2).mean() : data(data.size() / 2);
}

}  // namespace reprojection::feature_extraction