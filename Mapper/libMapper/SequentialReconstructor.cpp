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
    Eigen::Matrix4d SequentialReconstructor::chooseInitialPair(int& imgIdx1, int& imgIdx2)
    {
        // sort matched images by number of matches:
        std::vector<std::pair<std::pair<int, int>, std::vector<Match>>> featureMatchesVec;
        for(const auto& match : featureMatches)
        {
            featureMatchesVec.push_back(match);
        }

        // sort by value size
        std::sort(featureMatchesVec.begin(), featureMatchesVec.end(), sortMatches);

        imgIdx1 = featureMatchesVec[0].first.first;
        imgIdx2 = featureMatchesVec[0].first.second;

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

        Eigen::Matrix4d relativePose = Eigen::Matrix4d::Identity();
        essentialMatToPose(essentialMat, features1, features2, intrinsics1, intrinsics2, relativePose);

        return relativePose;
    }

    void triangulate2View(const double x1, const double y1,
                          const double x2, const double y2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          Eigen::Vector3d& landmark)
    {
        std::cout << "pose1: " << pose1 << std::endl;
        std::cout << "pose2: " << pose2 << std::endl;
        auto row1 = x1 * pose1.row(2) - pose1.row(0);
        auto row2 = y1 * pose1.row(2) - pose1.row(1);
        auto row3 = x2 * pose2.row(2) - pose2.row(0);
        auto row4 = y2 * pose2.row(2) - pose2.row(1);

        Eigen::Matrix4d A;
        A.row(0) = row1;
        A.row(1) = row2;
        A.row(2) = row3;
        A.row(3) = row4;

        std::cout << "A: " << A << std::endl;

        // Eigen::FullPivLU<Eigen::Matrix4d> lu_decomp(A);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    

        landmark = svd.matrixV().col(3).hnormalized();

        // auto landmarkHomogeneous = lu_decomp.kernel();

        // std::cout << "landmarkHomogeneous: " << landmarkHomogeneous << std::endl;
    }

    void SequentialReconstructor::triangulateInitialPair(const Eigen::Matrix4d relativePose,
                                                         const int imgIdx1,
                                                         const int imgIdx2)
    {
        auto matchedFeatIds = featureMatches[std::make_pair(imgIdx1, imgIdx2)];

        for(const auto& matchedPair : matchedFeatIds)
        {
            auto featPtr1 = features[imgIdx1][matchedPair.idx1];
            auto featPtr2 = features[imgIdx2][matchedPair.idx2];

            // project intrinsics into image plane:
            // featPtr1->featCoord.x

            auto intrinsics1 = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                                 imgShapes[imgIdx1].first,
                                                                 imgShapes[imgIdx1].second,
                                                                 defaultFov);

            // u = f_x * x + c_x 
            double x1 = (featPtr1->featCoord.x - intrinsics1(0,2)) / intrinsics1(0,0);
            double y1 = (featPtr1->featCoord.y - intrinsics1(1,2)) / intrinsics1(1,1);

            auto intrinsics2 = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                                    imgShapes[imgIdx2].first,
                                                                    imgShapes[imgIdx2].second,
                                                                    defaultFov);

            double x2 = (featPtr2->featCoord.x - intrinsics2(0,2)) / intrinsics2(0,0);
            double y2 = (featPtr2->featCoord.y - intrinsics2(1,2)) / intrinsics2(1,1);

            Eigen::Vector3d landmarkCoords;
            triangulate2View(x1, y1,
                             x2, y2,
                             Eigen::Matrix4d::Identity(),
                             relativePose,
                             landmarkCoords);

            Landmark landmarkCurr(landmarkCoords(0),landmarkCoords(1), landmarkCoords(2));
            TriangulatedFeature triangulatedFeature1(imgIdx1, matchedPair.idx1);
            TriangulatedFeature triangulatedFeature2(imgIdx2, matchedPair.idx2);
            landmarkCurr.triangulatedFeatures.push_back(triangulatedFeature1);
            landmarkCurr.triangulatedFeatures.push_back(triangulatedFeature2);
            
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

        registeredImages.resize(imgIds2Paths.size());
        std::fill(registeredImages.begin(), registeredImages.end(), false);

        cameraPoses.resize(imgIds2Paths.size());


        detectFeatures();

        matchImages();

        matchFeatures();

        int imgIdx1, imgIdx2;
        auto initialPose = chooseInitialPair(imgIdx1, imgIdx2);

        cameraPoses[imgIdx1] = Eigen::Matrix4d::Identity();
        cameraPoses[imgIdx2] = initialPose;
        registeredImages[imgIdx1] = true;
        registeredImages[imgIdx2] = true;

        triangulateInitialPair(initialPose, imgIdx1, imgIdx2);

        // add rest of views via solvepnp

    }


}
