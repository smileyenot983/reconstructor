#pragma once

#include <unordered_map>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "datatypes.h"
#include "utils.h"
#include "SequentialReconstructor.h"


namespace reconstructor::Core
{
  

struct ReprojectionError
{
    ReprojectionError(double observed_u,
                      double observed_v)
    : observed_u(observed_u)
    , observed_v(observed_v) 
    {}

    // [u,v] = intrinsics * extrinsics * [X, Y, Z]
    // with extrinsics(landmark in camera frame): p_c = R * p_w + t

    template <typename T>
    bool operator()(const T* cameraPose,
                    const T* cameraIntrinsics,
                    const T* landmarkWorldFrame,
                    T* residuals) const
    {
        // 1. apply extrinsics(3rot + 3trans):
        T landmarkCamFrame[3];
        ceres::AngleAxisRotatePoint(cameraPose, landmarkWorldFrame, landmarkCamFrame);
        landmarkCamFrame[0] += cameraPose[3];
        landmarkCamFrame[1] += cameraPose[4];
        landmarkCamFrame[2] += cameraPose[5];

        // 2. normalized on image plane(z=1)
        T landmarkImagePlaneX = landmarkCamFrame[0] /= landmarkCamFrame[2];
        T landmarkImagePlaneY = landmarkCamFrame[1] /= landmarkCamFrame[2];

        // 2. apply intrinsics(fx,fy,cx,cy)

        T predicted_u = cameraIntrinsics[0] * landmarkCamFrame[0] + cameraIntrinsics[1];
        T predicted_v = cameraIntrinsics[0] * landmarkCamFrame[1] + cameraIntrinsics[2];

        // calculate residuals
        residuals[0] = predicted_u - observed_u;
        residuals[1] = predicted_v - observed_v;

        return true;

    }

    static ceres::CostFunction* Create(const double observed_u,
                                       const double observed_v)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 3>(
                    new ReprojectionError(observed_u, 
                                          observed_v)));
    }

    double observed_u;
    double observed_v;
};


/* 
class which takes as input 2d - 3d matched coords, camera poses and intrinsics and adjusts 
3d points and camera poses(+ optionally intrinsics)
*/
class BundleAdjuster
{
public:
    // TODO: replace with map.find instead of [] to be able to pass const reference
    void adjust(std::unordered_map<int, std::vector<FeaturePtr<>>>& features,
                std::vector<Landmark>& landmarks,
                std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
                std::unordered_map<int, std::pair<int,int>> imgIdx2imgShape,
                std::vector<Eigen::Vector3d>& landmarksUpdated,
                std::vector<Eigen::Vector3d>& cameraPosesUpdated,
                double defaultFocalLengthPx = 500);

};

  
} // namespace reconstructor::Core
