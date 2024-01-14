#include "GeometricFilter.h"

#include <vector>

#include "utils.h"

using namespace reconstructor::Utils;
namespace reconstructor::Core
{
    Eigen::Matrix3d GeometricFilter::estimateEssential(const std::vector<FeaturePtr<>> &features1,
                                                       const std::vector<FeaturePtr<>> &features2,
                                                       const PinholeCamera &intrinsics1,
                                                       const PinholeCamera &intrinsics2,
                                                       std::vector<bool> &inlierMatchIds)
    {
        auto featuresCV1 = featuresToCvPoints(features1);
        auto featuresCV2 = featuresToCvPoints(features2);

        auto intrinsicsCV1 = intrinsics1.getMatrixCV();
        auto distortCV1 = intrinsics1.getDistortCV();

        auto intrinsicsCV2 = intrinsics2.getMatrixCV();
        auto distortCV2 = intrinsics2.getDistortCV();

        cv::Mat inliersCV;
        auto essentialMatCV = cv::findEssentialMat(featuresCV1,
                                                   featuresCV2,
                                                   intrinsicsCV1,
                                                   distortCV1,
                                                   intrinsicsCV2,
                                                   distortCV2);

        writeInliersToVector(inliersCV, inlierMatchIds);
        auto essentialMatEigen = cvMatToEigen3d(essentialMatCV);

        return essentialMatEigen;
    }

    Eigen::Matrix3d GeometricFilter::estimateFundamental(const std::vector<FeaturePtr<>> &features1,
                                                         const std::vector<FeaturePtr<>> &features2,
                                                         std::vector<bool> &inlierMatchIds)
    {
        auto featuresCV1 = featuresToCvPoints(features1);
        auto featuresCV2 = featuresToCvPoints(features2);

        cv::Mat inliersCV;
        auto fundamentalMat = cv::findFundamentalMat(featuresCV1, featuresCV2, inliersCV);

        // sometimes it fails to estimated fundamental mat
        if (fundamentalMat.empty())
        {
            return Eigen::Matrix3d::Zero();
        }
        else
        {
            writeInliersToVector(inliersCV, inlierMatchIds);
            auto eigenMat = cvMatToEigen3d(fundamentalMat);

            return eigenMat;
        }
    }
}
