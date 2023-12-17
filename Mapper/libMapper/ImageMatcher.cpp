#include "ImageMatcher.h"


namespace reconstructor::Core
{
    void FakeImgMatcher::match(const std::vector<std::pair<int, fs::path>>& imgIds2Paths,
                    const std::vector<std::vector<FeaturePtr<>>> features,
                    std::vector<std::vector<int>>& imgMatches)
    {
        int imgNum = imgIds2Paths.size();
        for(size_t imgId1 = 0; imgId1 < imgNum; ++imgId1)
        {
            std::vector<int> imgId1Matches;
            for(size_t imgId2 = 0; imgId2 < imgNum; ++imgId2)
            {
                imgId1Matches.push_back(imgId2);
            }

            imgMatches.push_back(imgId1Matches);
        }
    }
}