#include "BundleAdjuster.h"

#include <vector>
#include <unordered_map>

namespace reconstructor::Core
{

std::vector<Eigen::Vector3d> BundleAdjuster::adjust(std::unordered_map<int, std::vector<FeaturePtr<>>>& features,
                            std::vector<Landmark>& landmarks,
                            std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
                            std::unordered_map<int, std::pair<int,int>> imgIdx2imgShape,
                            double defaultFov,
                            double defaultFocalLengthmm)
{
    auto defaultImageHeight = 384.0;
    auto defaultImageWidth = 512.0;

    auto nTotalObservations = 0;
    auto nTotalCameras = imgIdx2camPose.size();
    auto nTotalLandmarks = landmarks.size();
    for(const auto& landmark : landmarks)
    {
        nTotalObservations += landmark.triangulatedFeatures.size();
    }

    // double* observations = new double(2 * nTotalObservations); 
    // double* extrinsic_params = new double(6 * nTotalCameras);
    // double* intrinsic_params = new double(3 * nTotalCameras); 
    // double* landmark_params = new double(3 * nTotalLandmarks); 
    double observations[2 * nTotalObservations];
    double extrinsic_params[6 * nTotalCameras];
    double intrinsic_params[3 * nTotalCameras];
    double landmark_params[3 * nTotalLandmarks];

    std::unordered_map<int, int> observation2Camera;
    std::unordered_map<int, int> observation2Landmark;

    for(const auto& [camIdx, camPose] : imgIdx2camPose)
    {
        std::cout << "camIdx: " << camIdx << std::endl;
        std::cout << "camPose: " << camPose << std::endl;

        Eigen::AngleAxisd angleAxis;
        angleAxis.fromRotationMatrix(camPose.block<3,3>(0,0));

        extrinsic_params[6 * camIdx] = angleAxis.axis()(0) * angleAxis.angle();
        extrinsic_params[6 * camIdx + 1] = angleAxis.axis()(1) * angleAxis.angle();
        extrinsic_params[6 * camIdx + 2] = angleAxis.axis()(2) * angleAxis.angle();
        extrinsic_params[6 * camIdx + 3] = camPose(0,3);
        extrinsic_params[6 * camIdx + 4] = camPose(1,3);
        extrinsic_params[6 * camIdx + 5] = camPose(2,3);

        auto intrinsics = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                            imgIdx2imgShape[camIdx].first,
                                            imgIdx2imgShape[camIdx].second,
                                            defaultFov);

        intrinsic_params[3 * camIdx] = intrinsics(0,0);
        intrinsic_params[3 * camIdx + 1] = intrinsics(0,2);
        intrinsic_params[3 * camIdx + 2] = intrinsics(1,2);
    }

    for(size_t i = 0; i < landmarks.size(); ++i)
    {
        landmark_params[3 * i] = landmarks[i].x;
        landmark_params[3 *i + 1] = landmarks[i].y;
        landmark_params[3 * i + 2] = landmarks[i].z;
    }
    

    ceres::Problem problem;
    int lastObservation = 0;
    for(size_t landmarkId = 0; landmarkId < landmarks.size(); ++landmarkId)
    {
        for(size_t featId = 0; featId < landmarks[landmarkId].triangulatedFeatures.size(); ++featId)
        {
            auto imgIdx = landmarks[landmarkId].triangulatedFeatures[featId].imgIdx;
            auto featIdx = landmarks[landmarkId].triangulatedFeatures[featId].featIdx;

            // observation2Landmark[lastObservation] = landmarkId;
            // observation2Camera[lastObservation] = imgIdx;
            
            observations[2 * lastObservation] = features[imgIdx][featIdx]->featCoord.x;
            observations[2 * lastObservation + 1] = features[imgIdx][featIdx]->featCoord.y;

            ceres::CostFunction* cost_function = ReprojectionError::Create(observations[2 * lastObservation],
                                                                           observations[2 * lastObservation + 1]);


            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     extrinsic_params + imgIdx * 6,
                                     intrinsic_params + imgIdx * 3,
                                     landmark_params + landmarkId * 3);

            ++lastObservation;
        }
    }

    std::cout << "before opt. extrinsicsArray[0]: " << extrinsic_params[0] << std::endl;
    std::cout << "before opt. extrinsicsArray[1]: " << extrinsic_params[1] << std::endl;
    std::cout << "before opt. extrinsicsArray[2]: " << extrinsic_params[2] << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 300;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    std::cout << "after opt. extrinsicsArray[0]: " << extrinsic_params[0] << std::endl;
    std::cout << "after opt. extrinsicsArray[1]: " << extrinsic_params[1] << std::endl;
    std::cout << "after opt. extrinsicsArray[2]: " << extrinsic_params[2] << std::endl;

    // update landmarks 
    std::vector<Eigen::Vector3d> updatedLandmarks;

    for(size_t i = 0; i < nTotalLandmarks; ++i)
    {
        Eigen::Vector3d updLandmark(landmark_params[3 * i],
                                    landmark_params[3 * i + 1],
                                    landmark_params[3 * i + 2]);

        updatedLandmarks.push_back(updLandmark);
    }

    return updatedLandmarks;
}




}