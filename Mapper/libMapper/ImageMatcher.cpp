#include "ImageMatcher.h"


namespace reconstructor::Core
{
    void FakeImgMatcher::match(const std::unordered_map<int, fs::path>& imgIds2Paths,
                               const std::unordered_map<int, std::vector<FeaturePtr<>>>& features,
                               std::unordered_map<int, std::vector<int>>& imgMatches)
    {
        int imgNum = imgIds2Paths.size();
        for(const auto& [imgId1, feats1] : features)
        {
            std::vector<int> imgId1Matches;
            for(const auto& [imgId2, feats2] : features)
            {
                if(imgId1 != imgId2)
                {
                    imgId1Matches.push_back(imgId2);
                }
                
            }
            imgMatches[imgId1] = imgId1Matches;
        }
    }
}