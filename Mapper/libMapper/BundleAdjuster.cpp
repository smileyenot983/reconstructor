#include "BundleAdjuster.h"

#include <vector>
#include <unordered_map>

#include "Camera.h"

namespace reconstructor::Core
{

std::unordered_map<int, int> BundleAdjuster::adjust(std::unordered_map<int, std::vector<FeaturePtr<>>>& features,
                                                    std::vector<Landmark>& landmarks,
                                                    std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
                                                    std::unordered_map<int, PinholeCamera> imgIdx2camIntrinsics,
                                                    std::vector<int> imgIdxOrder)
{
    auto nTotalObservations = 0;
    auto nTotalCameras = imgIdx2camPose.size();
    auto nTotalLandmarks = landmarks.size();
    for(const auto& landmark : landmarks)
    {
        nTotalObservations += landmark.triangulatedFeatures.size();
    }

    double observations[2 * nTotalObservations];
    double extrinsic_params[6 * nTotalCameras];
    double intrinsic_params[6 * nTotalCameras];
    double landmark_params[3 * nTotalLandmarks];

    // std::unordered_map<int, int> observation2Camera;
    // std::unordered_map<int, int> observation2Landmark;


    std::unordered_map<int, int> imgIdxGlobal2Local;
    std::unordered_map<int, int> imgIdxLocal2Global;
    int imgIdxLocal = 0;

    for(const auto& imgIdx : imgIdxOrder)
    {
        // auto intrinsics = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthPx,
        //                                                         imgIdx2imgShape[imgIdx].first,
        //                                                         imgIdx2imgShape[imgIdx].second);

        auto intrinsics = imgIdx2camIntrinsics[imgIdx];

        intrinsic_params[6 * imgIdxLocal + 0] = intrinsics.fX;
        intrinsic_params[6 * imgIdxLocal + 1] = intrinsics.fY;
        intrinsic_params[6 * imgIdxLocal + 2] = intrinsics.cX;
        intrinsic_params[6 * imgIdxLocal + 3] = intrinsics.cY;
        intrinsic_params[6 * imgIdxLocal + 4] = intrinsics.k1;
        intrinsic_params[6 * imgIdxLocal + 5] = intrinsics.k2;

        std::cout << "imgIdx: " << imgIdx << std::endl;

        auto camPose = imgIdx2camPose[imgIdx];

        Eigen::AngleAxisd angleAxis;
        angleAxis.fromRotationMatrix(camPose.block<3,3>(0,0));

        extrinsic_params[6 * imgIdxLocal] = angleAxis.axis()(0) * angleAxis.angle();
        extrinsic_params[6 * imgIdxLocal + 1] = angleAxis.axis()(1) * angleAxis.angle();
        extrinsic_params[6 * imgIdxLocal + 2] = angleAxis.axis()(2) * angleAxis.angle();
        extrinsic_params[6 * imgIdxLocal + 3] = camPose(0,3);
        extrinsic_params[6 * imgIdxLocal + 4] = camPose(1,3);
        extrinsic_params[6 * imgIdxLocal + 5] = camPose(2,3);

        imgIdxGlobal2Local[imgIdx] = imgIdxLocal;
        imgIdxLocal2Global[imgIdxLocal] = imgIdx;

        ++imgIdxLocal; 
    }

    for(size_t i = 0; i < landmarks.size(); ++i)
    {
        landmark_params[3 * i + 0] = landmarks[i].x;
        landmark_params[3 * i + 1] = landmarks[i].y;
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

            auto imgIdxLocal = imgIdxGlobal2Local[imgIdx];

            // observation2Landmark[lastObservation] = landmarkId;
            // observation2Camera[lastObservation] = imgIdx;
            
            observations[2 * lastObservation] = features[imgIdx][featIdx]->featCoord.x;
            observations[2 * lastObservation + 1] = features[imgIdx][featIdx]->featCoord.y;

            ceres::CostFunction* cost_function = ReprojectionError::Create(observations[2 * lastObservation],
                                                                           observations[2 * lastObservation + 1]);

            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     extrinsic_params + imgIdxLocal * 6,
                                     intrinsic_params + imgIdxLocal * 6,
                                     landmark_params + landmarkId * 3);

            ++lastObservation;
        }
    }

    // fix first camera
    double* extrinsic_params0 = extrinsic_params ;
    problem.SetParameterBlockConstant(extrinsic_params0);

    // fix second camera translation
    double* extrinsic_params1 = extrinsic_params + 6;
    problem.SetParameterization(extrinsic_params1, new ceres::SubsetParameterization(6, {3, 4, 5}));


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // update input info:
    for(size_t i = 0; i < landmarks.size(); ++i)
    {
        landmarks[i].x = landmark_params[3 * i + 0];
        landmarks[i].y = landmark_params[3 * i + 1];
        landmarks[i].z = landmark_params[3 * i + 2]; 
    }

    for(const auto& [localIdx, globalIdx] : imgIdxLocal2Global)
    {
        std::cout << "localIdx: " << localIdx
                  << "| globalIdx: " << globalIdx << std::endl;

        std::cout << "extrinsic_params[6 * localIdx + 0]: " << extrinsic_params[6 * localIdx + 0]
                  << "| extrinsic_params[6 * localIdx + 1]: " << extrinsic_params[6 * localIdx + 1]
                  << "| extrinsic_params[6 * localIdx + 2]: " << extrinsic_params[6 * localIdx + 2] << std::endl;

        // convert angle axis to rotation matrix
        double rotAngle = sqrt(extrinsic_params[6 * localIdx + 0] * extrinsic_params[6 * localIdx + 0]
                        + extrinsic_params[6 * localIdx + 1] * extrinsic_params[6 * localIdx + 1]
                        + extrinsic_params[6 * localIdx + 2] * extrinsic_params[6 * localIdx + 2]);

        Eigen::Vector3d rotAxis(extrinsic_params[6 * localIdx + 0] / (rotAngle + 1e-6),
                                extrinsic_params[6 * localIdx + 1] / (rotAngle + 1e-6),
                                extrinsic_params[6 * localIdx + 2] / (rotAngle + 1e-6));

        Eigen::AngleAxisd extrinsicsR(rotAngle, rotAxis); 

        Eigen::Matrix4d camPoseUpdated = Eigen::Matrix4d::Identity();
        camPoseUpdated.block<3,3>(0,0) = extrinsicsR.toRotationMatrix();
        camPoseUpdated(0,3) = extrinsic_params[6 * localIdx + 3];
        camPoseUpdated(1,3) = extrinsic_params[6 * localIdx + 4];
        camPoseUpdated(2,3) = extrinsic_params[6 * localIdx + 5];

        // auto intrinsics = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthPx,
        //                                                         imgIdx2imgShape[globalIdx].first,
        //                                                         imgIdx2imgShape[globalIdx].second);

        auto& intrinsics = imgIdx2camIntrinsics[globalIdx];
        intrinsics.fX = intrinsic_params[6 * localIdx + 0];
        intrinsics.fY = intrinsic_params[6 * localIdx + 1];
        intrinsics.cX = intrinsic_params[6 * localIdx + 2];
        intrinsics.cY = intrinsic_params[6 * localIdx + 3];
        intrinsics.k1 = intrinsic_params[6 * localIdx + 4];
        intrinsics.k2 = intrinsic_params[6 * localIdx + 5];

        std::cout << "pose before: " << imgIdx2camPose[globalIdx] << std::endl;
        std::cout << "pose after: " << camPoseUpdated << std::endl;

        imgIdx2camPose[globalIdx] = camPoseUpdated; 
    }

    return imgIdxGlobal2Local;

}




}