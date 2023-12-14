#pragma once

#include "FeatureMatcher.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace reconstructor::Core
{
    class FeatureMatcherSuperglue : public FeatureMatcher
    {
    public:
        FeatureMatcherSuperglue(const int imgHeight = 128,
                                const int imgWidth = 128,
                                const std::string& networkPath = "../models/superglue_model.zip"); 

        void matchFeatures(const std::vector<FeaturePtr<>>& features1,
                           const std::vector<FeaturePtr<>>& features2,
                           std::vector<Match>& matches) override;

    private:
        torch::jit::script::Module superGlue;

        const int imgHeight;
        const int imgWidth;

    };
}

