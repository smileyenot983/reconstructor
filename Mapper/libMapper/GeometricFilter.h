#pragma once

#include "datatypes.h"

namespace reconstructor::Core
{
    // class GeometricFilter
    // {
    //     public:
    //         GeometricFilter(int sampleSize){}

    //         /*

    //         */
    //         virtual void fit() = 0;

    // };

    // filters keypoint matches based on epipolar geometry: F : fundamental matrix(uncalibrated)
    //                                                      E : essential matrix(calibrated)
    // it must hold that matched features have x'^TFx = 0 and x'^TEx = 0
    class GeometricFilter
    {
    public:
        // estimates essential matrix(in case intrinsics are known) with 5pt algorithm
        cv::Mat estimateEssentialMat(const std::vector<FeaturePtr<>> &features1,
                                             const std::vector<FeaturePtr<>> &features2,
                                             const Eigen::Matrix3d &intrinsics1,
                                             const Eigen::Matrix3d &intrinsics2);

        // estimates fundamental matrix(in cast intrinsics are not known)
        cv::Mat estimateFundamentalMat(const std::vector<FeaturePtr<>> &features1,
                                               const std::vector<FeaturePtr<>> &features2);

        // applies given geometric filter
        void filterFeatures(const std::vector<FeaturePtr<>> &features1,
                            const std::vector<FeaturePtr<>> &features2,
                            const std::vector<Match> &featMatches,
                            std::vector<Match> &featMatchesFiltered,
                            std::shared_ptr<Eigen::Matrix3d> intrinsics1 = nullptr,
                            std::shared_ptr<Eigen::Matrix3d> intrinsics2 = nullptr);
    };
}