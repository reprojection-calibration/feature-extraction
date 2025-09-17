#include "feature_extraction/optimizer_handler.hpp"

namespace reprojection_calibration::feature_extraction {

void OptimizerHandler(double const data, CameraParameters* const camera_parameters,
                      std::function<ceres::CostFunction*(double)> cost_function_factory) {
    ceres::Problem problem;
    ceres::CostFunction* cost_function{cost_function_factory(data)};
    problem.AddResidualBlock(cost_function, nullptr, camera_parameters->GetParameterPtr());

    ceres::Solver::Options const options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

}  // namespace reprojection_calibration::feature_extraction