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

                // if this pair's feature were already matched(for features: (1,2)=(2,1) ), don't match again
                std::pair<int, int> inversePair(imgId2, imgId1);
                if(featureMatches.find(inversePair) != featureMatches.end())
                {
                    featureMatches[curPair] = featureMatches[inversePair];
                    continue;
                }

                featMatcher->matchFeatures(features1, features2, curMatches, imgShapes[imgId1], imgShapes[imgId2]);
                

                // write mathes into vectors
                std::vector<FeaturePtr<>> featuresMatched1;
                std::vector<FeaturePtr<>> featuresMatched2;

                for(const auto& matchedFeatPair : curMatches)
                {
                    featuresMatched1.push_back(features1[matchedFeatPair.idx1]);
                    featuresMatched2.push_back(features2[matchedFeatPair.idx2]);
                }

                // apply geometric filter and leave only valid matches
                std::shared_ptr<std::vector<bool>> featMatchInliers = std::make_shared<std::vector<bool>>();
                featFilter->estimateFundamental(featuresMatched1, featuresMatched2, featMatchInliers);

                // write only filtered matches
                std::vector<Match> curMatchesFiltered;

                for(size_t matchIdx = 0; matchIdx < featMatchInliers->size(); ++matchIdx)
                {
                    // if true -> valid match
                    if(featMatchInliers->at(matchIdx))
                    {
                        curMatchesFiltered.push_back(curMatches[matchIdx]);
                    }
                }

                // std::cout << "matchesInitial: " << curMatches.size() << "| matchesFiltered: " << curMatchesFiltered.size() << std::endl; 
                featureMatches[curPair] = curMatchesFiltered;
            }
        }
    }

    void essentialMatToPose(const Eigen::Matrix3d& essentialMat,
                             const std::vector<FeaturePtr<>>& features1,
                             const std::vector<FeaturePtr<>>& features2,
                             const Eigen::Matrix3d& intrinsics1,
                             const Eigen::Matrix3d& intrinsics2,
                             Eigen::Matrix4d& relativePose)
    {
        auto essentialMatCV = reconstructor::Utils::eigen3dToCVMat(essentialMat, CV_64F);

        auto featuresCV1 = reconstructor::Utils::featuresToCvPoints(features1);
        auto featuresCV2 = reconstructor::Utils::featuresToCvPoints(features2);

        auto intrinsicsCV1 = reconstructor::Utils::eigen3dToCVMat(intrinsics1);
        auto intrinsicsCV2 = reconstructor::Utils::eigen3dToCVMat(intrinsics2);

        cv::Mat R,t;
        cv::recoverPose(essentialMatCV, featuresCV1, featuresCV2, intrinsicsCV1, R, t);

        for(size_t row = 0; row < 3; ++row)
        {
            for(size_t col = 0; col < 3; ++col)
            {
                relativePose(row,col) = R.at<double>(row,col);
            }
            relativePose(row,3) = t.at<double>(row);
        }
        relativePose(3,3) = 1.0;
    }

    bool sortMatches(std::pair<std::pair<int, int>, std::vector<Match>> matches1,
                     std::pair<std::pair<int, int>, std::vector<Match>> matches2)
    {
        return matches1.second.size() > matches2.second.size();
    }


    // 1. find pair with highest number of matches
    Eigen::Matrix4d SequentialReconstructor::chooseInitialPair()
    {
        // sort matched images by number of matches:
        std::vector<std::pair<std::pair<int, int>, std::vector<Match>>> featureMatchesVec;
        for(const auto& match : featureMatches)
        {
            featureMatchesVec.push_back(match);
        }

        // sort by value size
        std::sort(featureMatchesVec.begin(), featureMatchesVec.end(), sortMatches);

        auto imgIdx1 = featureMatchesVec[0].first.first;
        auto imgIdx2 = featureMatchesVec[0].first.second;

        // backproject feature coords(K^{-1}*x)
        auto intrinsics1 = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                                 imgShapes[imgIdx1].first,
                                                                 imgShapes[imgIdx1].second,
                                                                 defaultFov);

        auto intrinsics2 = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                                 imgShapes[imgIdx2].first,
                                                                 imgShapes[imgIdx2].second,
                                                                 defaultFov);

        // extract matches between given images:
        auto imgPair = std::make_pair(imgIdx1, imgIdx2);
        auto featMatches = featureMatches[imgPair];      
        std::vector<FeaturePtr<>> features1;
        std::vector<FeaturePtr<>> features2;

        for(const auto& matchedFeatPair : featMatches)
        {
            // feat
            features1.push_back(features[imgIdx1][matchedFeatPair.idx1]);
            features2.push_back(features[imgIdx2][matchedFeatPair.idx2]);
            // features2.push_back()
        }
        
        auto essentialMat = featFilter->estimateEssential(features1, features2, intrinsics1, intrinsics2);

        std::cout << "essentialMat: " << essentialMat << std::endl;

        Eigen::Matrix4d relativePose;
        essentialMatToPose(essentialMat, features1, features2, intrinsics1, intrinsics2, relativePose);

        return relativePose;

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

        auto initialPose = chooseInitialPair();

        

        // add rest of views via solvepnp

    }


}
