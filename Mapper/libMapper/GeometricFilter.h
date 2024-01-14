#pragma once

#include "datatypes.h"
#include "Camera.h"

namespace reconstructor::Core
{

    /*
       Filters keypoint matches based on epipolar geometry: F : fundamental matrix(uncalibrated)
                                                            E : essential matrix(calibrated)
       it must hold that matched features have x'^TFx = 0 and x'^TEx = 0
    /*
    class GeometricFilter
    {
    public:

        /*
            Estimates essential matrix and corresponding inliers
            (mapping on mm level (K'^{-1} [u' v' 1])^T * E * (K^{-1} [u v 1])) )
        */
    Eigen::Matrix3d estimateEssential(const std::vector<FeaturePtr<>> &features1,
                                      const std::vector<FeaturePtr<>> &features2,
                                      const PinholeCamera &intrinsics1,
                                      const PinholeCamera &intrinsics2,
                                      std::vector<bool> &inlierMatchIds);

    /*
        Estimates fundamental matrix and corresponding inliers
        (mapping on pixel level  ([u' v' 1])^T * F * ([u v 1]))
    */
    Eigen::Matrix3d estimateFundamental(const std::vector<FeaturePtr<>> &features1,
                                        const std::vector<FeaturePtr<>> &features2,
                                        std::vector<bool> &inlierMatchIds);

};
}