#include "FeatureMatcherSuperglue.h"

#include <algorithm>


namespace reconstructor::Core
{
    namespace
    {
        /*
        normalizes keypoints to make sure their locations are within
        desired range
        */
        void normalizeKeypoints(const std::vector<FeaturePtr<>>& keypoints,
                                const int imgHeight,
                                const int imgWidth)
        {
            auto imgCenterX = imgWidth / 2;
            auto imgCenterY = imgHeight / 2;
            auto imgScale = std::max(imgHeight, imgWidth) * 0.7; 

            for(size_t i = 0; i < keypoints.size(); ++i)
            {
                // auto scaling
                // keypoints[i].featCoord.x = (keypoints[i].featCoord.x - imgWidth/2);
            }
        }
    }

    FeatureMatcherSuperglue::FeatureMatcherSuperglue(const std::string& networkPath)
    {
        try
        {
            superGlue = torch::jit::load(networkPath);
        }
        catch (const c10::Error &e)
        {
            std::cerr << "erro loading superNet" << std::endl;
        }
    }

    void FeatureMatcherSuperglue::matchFeatures(const std::vector<FeaturePtr<>>& features1,
                                                const std::vector<FeaturePtr<>>& features2,
                                                std::vector<Match>& matches)
    {
        // normalizeKeypoints()
    }
}