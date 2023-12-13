#pragma once

#include "FeatureMatcher.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace reconstructor::Core
{
    class FeatureMatcherSuperglue : public FeatureMatcher
    {
    public:
        FeatureMatcherSuperglue(const std::string& networkPath,
                                const int imgHeight,
                                const int imgWidth);

        void matchFeatures(const std::vector<FeaturePtr<>>& features1,
                           const std::vector<FeaturePtr<>>& features2,
                           std::vector<Match>& matches) override;

    private:
        torch::jit::script::Module superGlue;

        const int imgHeight;
        const int imgWidth;

    };
}

