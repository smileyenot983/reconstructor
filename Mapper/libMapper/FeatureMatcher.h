#pragma once

#include "opencv2/features2d.hpp"

#include "datatypes.h"

namespace reconstructor::Core
{

    struct Match
    {
    public:
        Match(size_t idx1, size_t idx2)
            : idx1(idx1), idx2(idx2)
        {
        }
        size_t idx1;
        size_t idx2;
    };

    class FeatureMatcher
    {
    public:
        virtual void matchFeatures(const std::vector<FeaturePtr<>> &features1,
                                   const std::vector<FeaturePtr<>> &features2,
                                   std::vector<Match> &matches) = 0;

        virtual ~FeatureMatcher() {}

    };

    class FlannMatcher : public FeatureMatcher
    {
    public:
        FlannMatcher();

        void matchFeatures(const std::vector<FeaturePtr<>> &features1,
                           const std::vector<FeaturePtr<>> &features2,
                           std::vector<Match>& matches) override;

    private:
        cv::Ptr<cv::DescriptorMatcher> matcher;
        const float ratioThresh = 0.7;
    };

} // namespace reconstructor::Core