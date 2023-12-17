#include "GeometricFilter.h"
#include <vector>

namespace reconstructor::Core
{
    std::vector<cv::Point2f> featuresToCV(const std::vector<FeaturePtr<>>& features)
    {
        std::vector<cv::Point2f> featuresCV(features.size());

        for(size_t featIdx = 0; featIdx < features.size(); ++featIdx)
        {
            cv::Point2f pt(features[featIdx]->featCoord.x, features[featIdx]->featCoord.y);
            featuresCV[featIdx] = pt;

        }
        return featuresCV;
    }

    cv::Mat eigen3dToCVMat(const Eigen::Matrix3d& eigenMat)
    {
        cv::Mat cvMat;
        for(size_t row=0; row < 3; ++row)
        {
            for(size_t col=0; col < 3; ++col)
            {
                cvMat.at<double>(row,col) = eigenMat(row,col);
            }
        }
        return cvMat;
    }

    Eigen::Matrix3d cvMatToEigen3d(const cv::Mat& cvMat)
    {
        Eigen::Matrix3d eigenMat;
        for(size_t row=0; row < 3; ++row)
        {
            for(size_t col=0; col < 3; ++col)
            {
                eigenMat(row, col) = cvMat.at<double>(row,col);
            }
        }
        return eigenMat;

    }

    cv::Mat GeometricFilter::estimateEssentialMat(const std::vector<FeaturePtr<>>& features1,
                                                            const std::vector<FeaturePtr<>>& features2,
                                                            const Eigen::Matrix3d& intrinsics1,
                                                            const Eigen::Matrix3d& intrinsics2)
    {
        auto featuresCV1 = featuresToCV(features1);
        auto featuresCV2 = featuresToCV(features2);

        auto intrinsicsCV1 = eigen3dToCVMat(intrinsics1);
        auto intrinsicsCV2 = eigen3dToCVMat(intrinsics2);

        // does not yet support different intrinsics
        // TODO: use cv::undistort and support different intrinsics
        if(intrinsics1 != intrinsics2)
        {
            throw std::runtime_error("Different intrinsics are not yet supported");
        }

        cv::Mat inliers;
        auto essentialMat = cv::findEssentialMat(featuresCV1,
                                                 featuresCV2,
                                                 intrinsicsCV1,
                                                 cv::RANSAC,
                                                 0.999,
                                                 1.0,
                                                 inliers);

        auto eigenMat = cvMatToEigen3d(essentialMat);


        return inliers;
    }

    cv::Mat GeometricFilter::estimateFundamentalMat(const std::vector<FeaturePtr<>> &features1,
                                                            const std::vector<FeaturePtr<>> &features2)
    {
        // first convert matched features to cv compatible format:
        auto featuresCV1 = featuresToCV(features1);
        auto featuresCV2 = featuresToCV(features2);

        cv::Mat inliers;
        auto fundamentalMat = cv::findFundamentalMat(featuresCV1, featuresCV2, inliers);

        auto eigenMat = cvMatToEigen3d(fundamentalMat);

        return inliers;
    }

    void GeometricFilter::filterFeatures(const std::vector<FeaturePtr<>>& features1,
                                  const std::vector<FeaturePtr<>>& features2,
                                  const std::vector<Match>& featMatches,
                                  std::vector<Match>& featMatchesFiltered,
                                  const std::shared_ptr<Eigen::Matrix3d> intrinsics1,
                                  const std::shared_ptr<Eigen::Matrix3d> intrinsics2)
    {
        // extract matched features
        std::vector<FeaturePtr<>> featuresMatched1;
        std::vector<FeaturePtr<>> featuresMatched2;
        for(const auto& matchedPair : featMatches)
        {
            featuresMatched1.push_back(features1[matchedPair.idx1]);
            featuresMatched2.push_back(features2[matchedPair.idx2]);

        }

        cv::Mat inlierIds;
        if(intrinsics1 && intrinsics2)
        {
            inlierIds = estimateEssentialMat(featuresMatched1,
                                                     featuresMatched2,
                                                     *intrinsics1,
                                                     *intrinsics2);
        }
        else
        {
            inlierIds = estimateFundamentalMat(featuresMatched1,
                                                         featuresMatched2);
        }

        for(size_t pairIdx = 0; pairIdx < inlierIds.rows; ++pairIdx)
        {
            if(inlierIds.at<uchar>(pairIdx) == 1)
            {
                featMatchesFiltered.push_back(featMatches[pairIdx]);
            }
        }
        // featMatchesFiltered

        // std::cout << "inlierIds.rows: " << inlierIds.rowBachelor’s de
    }

}
