#include "SequentialReconstructor.h"


#include "utils.h"


namespace reconstructor::Core
{

    SequentialReconstructor::SequentialReconstructor(const FeatDetectorType& featDetectorType,
                                                     const FeatMatcherType& featMatcherType)
    {
        // replace with Factory design pattern
        if(featDetectorType == FeatDetectorType::Orb)
        {
            featDetector = std::make_unique<FeatureORB>();
        }
        else if(featDetectorType == FeatDetectorType::SuperPoint)
        {
            featDetector = std::make_unique<FeatureSuperPoint>();
        }
        else
        {
            throw std::runtime_error("wrong feature detector type passed");
        }

        if(featMatcherType == FeatMatcherType::Flann)
        {
            featMatcher = std::make_unique<FlannMatcher>();
        }
        else if(featMatcherType == FeatMatcherType::SuperGlue)
        {
            featMatcher = std::make_unique<FeatureMatcherSuperglue>();
        }
        else
        {
            throw std::runtime_error("wrong feature matcher type passed");
        }
    }

    void SequentialReconstructor::detectFeatures()
    {
        for(const auto& [imgId, imgPath] : imgIds2Paths)
        {
            auto imgGray = reconstructor::Utils::readGrayImg(imgPath, imgMaxSize);
            auto imgPrepared = featDetector->prepImg(imgGray);

            std::vector<FeaturePtr<>> imgFeatures;
            featDetector->detect(imgPrepared, imgFeatures);

            features.push_back(imgFeatures);
        }
    }

    void SequentialReconstructor::matchImages()
    {

    }


    // void SequentialReconstructor::matchFeatures(const std::pair<int, int> matchedImagePairs)
    // {
    //     for(size_t i = 0; i < )
    // }

    void SequentialReconstructor::reconstruct(const std::string& imgFolder)
    {
        // extract all images from a given path and assign them ids
        int imgId = 0;
        for(const auto& entry : fs::directory_iterator(imgFolder))
        {
            auto imgId2PathPair = std::make_pair(imgId, entry.path());
            imgIds2Paths.push_back(imgId2PathPair);
            ++imgId;
        }

        detectFeatures();

        matchImages();

    }


}
