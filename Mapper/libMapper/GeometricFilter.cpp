#include "GeometricFilter.h"

#include <vector>

#include "utils.h"

using namespace reconstructor::Utils;
namespace reconstructor::Core
{
    Eigen::Matrix3d GeometricFilter::estimateEssential(const std::vector<FeaturePtr<>>& features1,
                                                    const std::vector<FeaturePtr<>>& features2,
                                                    const PinholeCamera& intrinsics1,
                                                    const PinholeCamera& intrinsics2,
                                                    std::shared_ptr<std::vector<bool>> inliers)
    {
        auto featuresCV1 = featuresToCvPoints(features1);
        auto featuresCV2 = featuresToCvPoints(features2);

        // if(intrinsics1 != intrinsics2)
        // {
        //     throw std::runtime_error("Different intrinsics are not yet supported");
        // }
        

        auto intrinsicsCV1 = intrinsics1.getMatrixCV();
        auto distortCV1 = intrinsics1.getDistortCV();

        auto intrinsicsCV2 = intrinsics2.getMatrixCV();
        auto distortCV2 = intrinsics2.getDistortCV();

        std::cout << " intrinsicsCV1: " << intrinsicsCV1 << std::endl;
        std::cout << " intrinsicsCV2: " << intrinsicsCV2 << std::endl;
        std::cout << " distortCV1: " << distortCV1 << std::endl;
        std::cout << " distortCV2: " << distortCV2 << std::endl;
        

        cv::Mat inliersCV;
        auto essentialMatCV = cv::findEssentialMat(featuresCV1,
                                                   featuresCV2,
                                                 intrinsicsCV1,
                                                 distortCV1,
                                                 intrinsicsCV2,
                                                 distortCV2);

        std::cout << "essentialMatCV.type(): " << essentialMatCV.type() << std::endl;

        if(inliers)
        {
            reconstructor::Utils::writeInliersToVector(inliersCV, *inliers);
        }        
        
        auto essentialMatEigen = cvMatToEigen3d(essentialMatCV);

        return essentialMatEigen;
    }

    Eigen::Matrix3d GeometricFilter::estimateFundamental(const std::vector<FeaturePtr<>>& features1,
                                                 const std::vector<FeaturePtr<>>& features2,
                                                 std::shared_ptr<std::vector<bool>> inliers)
    {
        auto featuresCV1 = featuresToCvPoints(features1);
        auto featuresCV2 = featuresToCvPoints(features2);

        cv::Mat inliersCV;
        auto fundamentalMat = cv::findFundamentalMat(featuresCV1, featuresCV2, inliersCV);

        writeInliersToVector(inliersCV, *inliers);
        auto eigenMat = cvMatToEigen3d(fundamentalMat);

        return eigenMat;
    }   


}
