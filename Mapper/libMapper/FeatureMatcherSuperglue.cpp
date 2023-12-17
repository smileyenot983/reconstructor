#include "FeatureMatcherSuperglue.h"

#include <algorithm>

#include "utils.h"


namespace reconstructor::Core
{
    namespace
    {
        std::vector<torch::jit::IValue> featsToTensors(const std::vector<FeaturePtr<double>>& features)
        {
            int featuresNum = static_cast<int>(features.size());
            int descSize = static_cast<int>(features[0]->featDesc.desc.size());
            
            torch::Tensor featCoords = torch::zeros({1, featuresNum, 2});
            torch::Tensor featDescs = torch::zeros({1, descSize, featuresNum});
            torch::Tensor featScores = torch::zeros({1, featuresNum});

        
            for(int featIdx = 0; featIdx < features.size(); ++featIdx)
            {
                FeatureConfPtr<double> featureConf = std::dynamic_pointer_cast<FeatureConf<double>>(features[featIdx]);

                featCoords[0][featIdx][0] = featureConf->featCoord.x;
                featCoords[0][featIdx][1] = featureConf->featCoord.y;


                featDescs[0].index({torch::indexing::Slice(0, descSize), featIdx}) = torch::from_blob(featureConf->featDesc.desc.data(), 
                                                                                                         {descSize}, torch::kFloat32);    

                featScores[0][featIdx] = featureConf->conf;

            }

            std::vector<torch::jit::IValue> inputTensors = {featCoords, featDescs, featScores};
            return inputTensors;
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
            std::cerr << "error loading superNet" << std::endl;
        }
    }

    void FeatureMatcherSuperglue::matchFeatures(const std::vector<FeaturePtr<>>& features1,
                                                const std::vector<FeaturePtr<>>& features2,
                                                std::vector<Match>& matches,
                                                const std::pair<int, int> imgShape1,
                                                const std::pair<int, int> imgShape2)
    {
        // normalization will be performed every time, 
        auto featuresNormalized1 = reconstructor::Utils::normalizeFeatCoords(features1, imgShape1.first, imgShape1.second);
        auto featuresNormalized2 = reconstructor::Utils::normalizeFeatCoords(features2, imgShape2.first, imgShape2.second);

        auto tensorInputs0 = featsToTensors(featuresNormalized1);
        auto tensorInputs1 = featsToTensors(featuresNormalized2);

        // network expects: coords0, coords1, descs0, descs1, scores0, scores1
        std::vector<torch::jit::IValue> tensorInputs = {tensorInputs0[0], tensorInputs1[0], tensorInputs0[1],
                                                         tensorInputs1[1], tensorInputs0[2], tensorInputs1[2]};


        auto netOut = superGlue.forward(tensorInputs);

        // TODO: perform matches cross-check(not used in original paper, but could work)
        auto matches0 = netOut.toTuple()->elements()[0].toTensor();
        auto matches1 = netOut.toTuple()->elements()[1].toTensor();
        auto matchScores0 = netOut.toTuple()->elements()[2].toTensor();
        auto matchScores1 = netOut.toTuple()->elements()[3].toTensor();
        
        for(size_t featIdx = 0; featIdx < matches0.sizes()[1]; ++featIdx)
        {
            auto matchedFeatIdx = matches0[0][featIdx].item<int>();
            // -1 means no matches
            if(matchedFeatIdx != -1)
            {
                Match match(featIdx, matchedFeatIdx);
                matches.push_back(match);
            }

        }

    }
}