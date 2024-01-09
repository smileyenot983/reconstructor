#pragma once

#include <unordered_map>

#include "opencv2/features2d.hpp"

#include "datatypes.h"

namespace reconstructor::Core
{
    class FeatureMatcher
    {
    public:
        FeatureMatcher(const bool featNormalization = false)
        {}

        virtual void matchFeatures(const std::vector<FeaturePtr<>> &features1,
                                   const std::vector<FeaturePtr<>> &features2,
                                   std::map<int, int>& matches,
                                   const std::pair<int, int> imgShape1,
                                   const std::pair<int, int> imgShape2) = 0;


        virtual ~FeatureMatcher() {}

    protected:
    };

    class FlannMatcher : public FeatureMatcher
    {
    public:
        FlannMatcher();

        void matchFeatures(const std::vector<FeaturePtr<>> &features1,
                           const std::vector<FeaturePtr<>> &features2,
                           std::map<int, int>& matches,
                           const std::pair<int, int> imgShape1,
                            const std::pair<int, int> imgShape2) override;

    private:
        cv::Ptr<cv::DescriptorMatcher> matcher;
        const float ratioThresh = 0.7;
    };

} // namespace reconstructor::Core