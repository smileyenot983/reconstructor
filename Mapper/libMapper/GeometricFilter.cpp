#include "GeometricFilter.h"

#include <vector>

#include "utils.h"

using namespace reconstructor::Utils;
namespace reconstructor::Core
{

    void writeInliersToVector(const cv::Mat& inliersCV,
                              std::vector<bool>& inliersVec)
    {
        // std::cout << "inliersVec.rows: " << inliersCV.rows << std::endl;
        // std::cout << "inliersVec.cols: " << inliersCV.cols << std::endl;
        // std::cout << "inliersCV.at<uchar>(matchIdx): " << static_cast<unsigned>(inliersCV.at<uchar>(0)) << std::endl;
        // std::cout << "inliersCV.at<uchar>(matchIdx): " << static_cast<unsigned>(inliersCV.at<uchar>(0)) << std::endl;
        for(size_t matchIdx = 0; matchIdx < inliersCV.rows; ++matchIdx)
        {
            inliersVec.push_back(inliersCV.at<uchar>(matchIdx));
            // std::cout << "inliersCV.at<uchar>(matchIdx): " << inliersCV.at<uchar>(matchIdx) << std::endl;
        }
    }

    Eigen::Matrix3d GeometricFilter::estimateEssential(const std::vector<FeaturePtr<>>& features1,
                                                    const std::vector<FeaturePtr<>>& features2,
                                                    const Eigen::Matrix3d& intrinsics1,
                                                    const Eigen::Matrix3d& intrinsics2,
                                                    std::shared_ptr<std::vector<bool>> inliers)
    {
        auto featuresCV1 = featuresToCvPoints(features1);
        auto featuresCV2 = featuresToCvPoints(features2);

        if(intrinsics1 != intrinsics2)
        {
            throw std::runtime_error("Different intrinsics are not yet supported");
        }
        
        auto intrinsicsCV1 = eigen3dToCVMat(intrinsics1);
        auto intrinsicsCV2 = eigen3dToCVMat(intrinsics2);

        cv::Mat inliersCV;
        auto essentialMatCV = cv::findEssentialMat(featuresCV1,
                                                 featuresCV2,
                                                 intrinsicsCV1,
                                                 cv::RANSAC,
                                                 0.999,
                                                 1.0,
                                                 inliersCV);

        std::cout << "essentialMatCV.type(): " << essentialMatCV.type() << std::endl;

        if(inliers)
        {
            writeInliersToVector(inliersCV, *inliers);
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
