#include "SequentialReconstructor.h"


#include "utils.h"


namespace reconstructor::Core
{

    SequentialReconstructor::SequentialReconstructor(const FeatDetectorType& featDetectorType,
                                                     const FeatMatcherType& featMatcherType,
                                                     const ImgMatcherType& imgMatcherType)
    {
        // replace with Factory design pattern
        switch(featDetectorType)
        {
        case FeatDetectorType::Orb:
            featDetector = std::make_unique<FeatureORB>();
            break;
        case FeatDetectorType::SuperPoint:
            featDetector = std::make_unique<FeatureSuperPoint>();
            break;
        default:
            throw std::runtime_error("SequentialReconstructor. Wrong feature detector type passed");
        }

        switch (featMatcherType)
        {
        case FeatMatcherType::Flann:
            featMatcher = std::make_unique<FlannMatcher>();
            break;
        case FeatMatcherType::SuperGlue:
            featMatcher = std::make_unique<FeatureMatcherSuperglue>();
            break;
        default:
            throw std::runtime_error("SequentialReconstructor. Wrong feature matcher type passed");
        }

        switch (imgMatcherType)
        {
        case ImgMatcherType::Fake:
            imgMatcher = std::make_unique<FakeImgMatcher>();
            break;
        default:
            throw std::runtime_error("SequentialReconstructor. Wrong image matcher type passed");
        }

        featFilter = std::make_unique<GeometricFilter>();
    }

    void SequentialReconstructor::detectFeatures()
    {
        for(const auto& [imgId, imgPath] : imgIds2Paths)
        {
            auto imgGray = reconstructor::Utils::readGrayImg(imgPath, imgMaxSize);
            auto imgPrepared = featDetector->prepImg(imgGray);

            imgShapes.push_back(std::make_pair(imgPrepared.rows, imgPrepared.cols));

            std::vector<FeaturePtr<>> imgFeatures;
            featDetector->detect(imgPrepared, imgFeatures);

            features.push_back(imgFeatures);
        }
    }

    void SequentialReconstructor::matchImages()
    {
        imgMatcher->match(imgIds2Paths, features, imgMatches);
    }

    /*matches features between previously matched images*/
    void SequentialReconstructor::matchFeatures()
    {
        // std::map<std::pair<int, int>, std::vector<Match>> featureMatches;
        // std::vector<std::vector<FeaturePtr<>>> features;
        
        for(size_t imgId1 = 0; imgId1 < imgMatches.size(); ++imgId1)
        {
            for(size_t imgId2 = 0; imgId2 < imgMatches[imgId1].size(); ++imgId2)
            {
                if(imgId1 == imgId2)
                {
                    continue;
                }
                auto features1 = features[imgId1];
                auto features2 = features[imgId2];
                std::pair<int, int> curPair(imgId1, imgId2);

                std::vector<Match> curMatches;

                featMatcher->matchFeatures(features1, features2, curMatches, imgShapes[imgId1], imgShapes[imgId2]);

                std::cout << "curMatches.size(): " << curMatches.size() << std::endl;

                std::vector<Match> curMatchesFiltered;
                featFilter->filterFeatures(features1, features2, curMatches, curMatchesFiltered);

                std::cout << "curMatchesFiltered.size(): " << curMatchesFiltered.size() << std::endl;
                featureMatches[curPair] = curMatches;
            }
        }
    }

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

        matchFeatures();

    }


}
