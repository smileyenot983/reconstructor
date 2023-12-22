#pragma once

#include "datatypes.h"

namespace reconstructor::Core
{


    // filters keypoint matches based on epipolar geometry: F : fundamental matrix(uncalibrated)
    //                                                      E : essential matrix(calibrated)
    // it must hold that matched features have x'^TFx = 0 and x'^TEx = 0
    class GeometricFilter
    {
    public:


        // for now only equal intrinsics are supported
        Eigen::Matrix3d estimateEssential(const std::vector<FeaturePtr<>>& features1,
                                  const std::vector<FeaturePtr<>>& features2,
                                  const Eigen::Matrix3d& intrinsics1,
                                  const Eigen::Matrix3d& intrinsics2,
                                  std::shared_ptr<std::vector<bool>> inliers = nullptr);


        Eigen::Matrix3d estimateFundamental(const std::vector<FeaturePtr<>>& features1,
                                    const std::vector<FeaturePtr<>>& features2,
                                    std::shared_ptr<std::vector<bool>> inliers = nullptr);


    };
}