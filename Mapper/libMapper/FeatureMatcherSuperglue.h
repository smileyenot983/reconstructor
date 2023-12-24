#pragma once

#include "FeatureMatcher.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace reconstructor::Core
{
    class FeatureMatcherSuperglue : public FeatureMatcher
    {
    public:
        FeatureMatcherSuperglue(const std::string& networkPath = "../models/superglue_model.zip"); 

        void matchFeatures(const std::vector<FeaturePtr<>>& features1,
                           const std::vector<FeaturePtr<>>& features2,
                           std::unordered_map<int, int>& matches,
                           const std::pair<int, int> imgShape1,
                           const std::pair<int, int> imgShape2) override;

    private:
        torch::jit::script::Module superGlue;

    };
}

