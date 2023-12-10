#pragma once

#include "FeatureMatcher.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace reconstructor::Core
{
    class FeatureMatcherSuperglue : public FeatureMatcher
    {
    public:
        FeatureMatcherSuperglue(const std::string& networkPath);

        void matchFeatures(const std::vector<FeaturePtr<>>& features1,
                        const std::vector<FeaturePtr<>>& features2,
                        std::vector<Match>& matches) override;

        void setImageSizes();

    private:
        torch::jit::script::Module superGlue;

    };
}

