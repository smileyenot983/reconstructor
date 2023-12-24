#include "FeatureMatcher.h"

#include <iostream>

namespace reconstructor::Core
{

/*
writes the keypoint descriptor to cv::Mat
*/
cv::Mat featDescToCV(const std::vector<FeaturePtr<>>& features)
{
    auto descSize = features[0]->featDesc.desc.size();
    cv::Mat descriptorsCV(features.size(), descSize, features[0]->featDesc.type);

    for(size_t featIdx = 0; featIdx < features.size(); ++featIdx)
    {
        for(size_t descIdx = 0; descIdx < descSize; ++descIdx)
        {
            descriptorsCV.at<float>(featIdx, descIdx) = features[featIdx]->featDesc.desc[descIdx];
        }
    }

    return descriptorsCV;
}

FlannMatcher::FlannMatcher()
{
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void FlannMatcher::matchFeatures(const std::vector<FeaturePtr<>>& features1,
                                 const std::vector<FeaturePtr<>>& features2,
                                 std::unordered_map<int, int>& matches,
                                 const std::pair<int, int> imgShape1,
                                 const std::pair<int, int> imgShape2)
{
    // make sure there are descriptors
    assert(features1.size() > 0 && features2.size() > 0);
    // make sure descriptors have same type and length
    assert(features1[0]->featDesc.type == features1[0]->featDesc.type);
    assert(features1[0]->featDesc.desc.size() == features1[0]->featDesc.desc.size());

    // cv matcher wants descriptors to be structured as cv::Mat
    auto descriptorsCV1 = featDescToCV(features1);
    auto descriptorsCV2 = featDescToCV(features2);
    
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptorsCV1, descriptorsCV2, knnMatches, 2);


    // filter resulting matches(Lowe's ratio test):
    for(size_t i = 0; i < knnMatches.size(); ++i)
    {
        if(knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
        {
            matches[knnMatches[i][0].queryIdx] = knnMatches[i][0].trainIdx;
            // Match match(knnMatches[i][0].queryIdx, knnMatches[i][0].trainIdx);
            // matches.push_back(match);
            // goodMatches.push_back(knnMatches[i][0]);
        }
    }
}

} // namespace reconstructor::Core