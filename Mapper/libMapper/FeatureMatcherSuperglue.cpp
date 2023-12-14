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
        std::vector<FeaturePtr<double>> normalizeFeatCoords(const std::vector<FeaturePtr<>>& features,
                                                     const int imgHeight,
                                                     const int imgWidth)
        {   
            std::vector<FeaturePtr<double>> featuresNormalized;

            // scaling to have kpts in range [-0.7, 0.7]
            auto imgCenterX = imgWidth / 2; // = 480/2 = 240
            auto imgCenterY = imgHeight / 2; // = 640/2 = 320
            // casted to double, cause floating point coords required
            auto imgScale = static_cast<double>(std::max(imgHeight, imgWidth) * 0.7); // 640*0.7 = 448

            for(size_t i = 0; i < features.size(); ++i)
            {
                // copy into new normalized feature
                FeatureConfPtr<> featOriginal = std::dynamic_pointer_cast<FeatureConf<>>(features[i]);

                FeatCoord<double> coordNormalized;
                // [10, 20]
                coordNormalized.x = (featOriginal->featCoord.x - imgCenterX) / imgScale;
                coordNormalized.y = (featOriginal->featCoord.y - imgCenterY) / imgScale;
                
                FeaturePtr<double> featNormalized = std::make_shared<FeatureConf<double>>(coordNormalized,
                                                                              featOriginal->featDesc,
                                                                              featOriginal->conf);

                featuresNormalized.push_back(featNormalized);
            }

            return featuresNormalized;
        }

        std::vector<torch::jit::IValue> featsToTensors(const std::vector<FeaturePtr<double>>& features)
        {
            int featuresNum = static_cast<int>(features.size());
            int descSize = static_cast<int>(features[0]->featDesc.desc.size());
            
            torch::Tensor featCoords = torch::zeros({1, featuresNum, 2});
            torch::Tensor featDescs = torch::zeros({1, descSize, featuresNum});
            torch::Tensor featScores = torch::zeros({1, featuresNum});

            for(size_t featIdx = 0; featIdx < features.size(); ++featIdx)
            {
                FeatureConfPtr<double> featureConf = std::dynamic_pointer_cast<FeatureConf<double>>(features[featIdx]);

                featCoords[0][featIdx][0] = featureConf->featCoord.x;
                featCoords[0][featIdx][1] = featureConf->featCoord.y;


                featDescs[0].slice(1, 0, descSize) = torch::from_blob(featureConf->featDesc.desc.data(),
                                                                      {descSize, 1}, torch::kFloat32);

                featScores[0][featIdx] = featureConf->conf;

            }

            std::vector<torch::jit::IValue> inputTensors = {featCoords, featDescs, featScores};
            return inputTensors;
        }

    }

    FeatureMatcherSuperglue::FeatureMatcherSuperglue(const int imgHeight,
                                                     const int imgWidth,
                                                     const std::string& networkPath)
    : imgHeight(imgHeight)
    , imgWidth(imgWidth)
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
                                                std::vector<Match>& matches)
    {

        auto featuresNormalized1 = normalizeFeatCoords(features1, imgHeight, imgWidth);
        auto featuresNormalized2 = normalizeFeatCoords(features2, imgHeight, imgWidth);

        FeatureConfPtr<> featureConf1 = std::dynamic_pointer_cast<FeatureConf<>>(features1[0]);
        FeatureConfPtr<> featureConf2 = std::dynamic_pointer_cast<FeatureConf<>>(features1[1]);

        auto tensorInputs0 = featsToTensors(featuresNormalized1);
        auto tensorInputs1 = featsToTensors(featuresNormalized1);

        // network expects: coords0, coords1, descs0, descs1, scores0, scores1
        std::vector<torch::jit::IValue> tensorInputs = {tensorInputs0[0], tensorInputs1[0], tensorInputs0[1],
                                                         tensorInputs1[1], tensorInputs0[2], tensorInputs1[2]};

        auto netOut = superGlue.forward(tensorInputs);

        // TODO: perform matches cross-check(not used in original paper, but could work)
        auto matches0 = netOut.toTuple()->elements()[0].toTensor();
        auto matches1 = netOut.toTuple()->elements()[1].toTensor();
        auto matchScores0 = netOut.toTuple()->elements()[2].toTensor();
        auto matchScores1 = netOut.toTuple()->elements()[3].toTensor();

        std::cout << "matches0.sizes(): " << matches0.sizes() << std::endl;
        std::cout << "matches1.sizes(): " << matches1.sizes() << std::endl;
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