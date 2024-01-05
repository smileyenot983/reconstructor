#include "SequentialReconstructor.h"

#include <filesystem>

#include "utils.h"

namespace fs = std::filesystem;
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
        std::cout << "feature detection" << std::endl;
        for(const auto& [imgId, imgPath] : imgIds2Paths)
        {
            std::cout << "imgId: " << imgId << std::endl;
            // auto imgGray = reconstructor::Utils::readGrayImg(imgPath, imgMaxSize);
            auto imgOriginal = reconstructor::Utils::readImg(imgPath, imgMaxSize);
            cv::Mat imgGray;
            cv::cvtColor(imgOriginal, imgGray, cv::COLOR_RGB2GRAY);

            auto imgPrepared = featDetector->prepImg(imgGray);
            imgIdx2imgShape[imgId] = std::make_pair(imgPrepared.rows, imgPrepared.cols);

            std::vector<FeaturePtr<>> imgFeatures;
            featDetector->detect(imgPrepared, imgFeatures);

            // extract colors corresponding to each feature:
            for(auto& featPtr : imgFeatures)
            {
                auto featColor = imgOriginal.at<cv::Vec3b>(featPtr->featCoord.y, featPtr->featCoord.x);

                featPtr->featColor.red = featColor[0];
                featPtr->featColor.green = featColor[1];
                featPtr->featColor.blue = featColor[2];
            }

            // features.push_back(imgFeatures);
            features[imgId] = imgFeatures;
        }
    }

    Eigen::Matrix3d SequentialReconstructor::getIntrinsics(const int imgHeight,
                           const int imgWidth)
    {
        if(defaultFocalLengthPx != -1)
        {
            return reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthPx,
                                                   imgHeight,
                                                   imgWidth);
        }
        else
        {
            return reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                          imgHeight,
                                                          imgWidth,
                                                          defaultFov);
        }
    }


    void SequentialReconstructor::matchImages()
    {
        imgMatcher->match(imgIds2Paths, features, imgMatches);
    }

    void SequentialReconstructor::drawFeatMatchesAndSave(const std::string& outFolder)
    {
        // imgIds2Paths

        std::string matchesFolder = outFolder + "/" + "matches";
        fs::create_directories(matchesFolder);

        for(const auto& [imgPair, featMap] : featureMatches)
        {
            // std::cout << "imgIds2Paths[imgPair.first]" << imgIds2Paths[imgPair.first] << std::endl;
            // std::cout << "imgIds2Paths[imgPair.seconds]" << imgIds2Paths[imgPair.second] << std::endl;


            // read 2 images and merge them into single:
            auto imgPath1 = imgIds2Paths[imgPair.first];
            auto imgGray1 = reconstructor::Utils::readGrayImg(imgPath1, imgMaxSize);
            auto imgPrepared1 = featDetector->prepImg(imgGray1);

            auto imgPath2 = imgIds2Paths[imgPair.second];
            auto imgGray2 = reconstructor::Utils::readGrayImg(imgPath2, imgMaxSize);
            auto imgPrepared2 = featDetector->prepImg(imgGray2);

            auto imgType = imgPrepared1.type();
            cv::Mat imgMerged(std::max(imgPrepared1.rows, imgPrepared2.rows), imgPrepared1.cols + imgPrepared2.cols, imgType);

            // first put values from first image
            for(size_t row = 0; row < std::min(imgMerged.rows, imgPrepared1.rows); ++row)
            {
                for(size_t col = 0; col < imgPrepared1.cols; ++ col)
                {
                    imgMerged.at<float>(row,col) = 255 * imgPrepared1.at<float>(row,col);
                }
            }
            // now from second:
            int colOffset = imgPrepared1.cols;
            for(size_t row = 0; row < std::min(imgMerged.rows, imgPrepared2.rows); ++row)
            {
                for(size_t col = 0; col < imgPrepared2.cols; ++col)
                {
                    imgMerged.at<float>(row, col + colOffset) = 255 * imgPrepared2.at<float>(row, col);
                }
            }

            // extract matched features and draw them:
            auto featMatches = featureMatches[std::make_pair(imgPair.first, imgPair.second)];
            for(const auto& [featIdx1, featIdx2] : featMatches)
            {
                int x1 = features[imgPair.first][featIdx1]->featCoord.x;
                int y1 = features[imgPair.first][featIdx1]->featCoord.y;

                int x2 = features[imgPair.second][featIdx2]->featCoord.x + colOffset;
                int y2 = features[imgPair.second][featIdx2]->featCoord.y;
            
                cv::line(imgMerged, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0, 0, 255), 1);
            }

            // save resulting image
            auto imgFilename = matchesFolder + "/pair" + std::to_string(imgPair.first) + std::to_string(imgPair.second) + ".JPG";
            cv::imwrite(imgFilename, imgMerged);
        }
    }

    /*matches features between previously matched images*/
    void SequentialReconstructor::matchFeatures(bool filter)
    {
        // std::map<std::pair<int, int>, std::vector<Match>> featureMatches;
        // std::vector<std::vector<FeaturePtr<>>> features;
        
        std::cout << "matching features" << std::endl;
        for(const auto& [imgId1, imgMatches1] : imgMatches)
        {
            for(const auto& [imgId2, imgMatches2] : imgMatches)
            {
                if(imgId1 == imgId2)
                {
                    continue;
                }

                auto features1 = features[imgId1];
                auto features2 = features[imgId2];
                std::pair<int, int> curPair(imgId1, imgId2);
                // std::vector<Match> curMatches;
                std::unordered_map<int, int> curMatches;

                // if this pair's feature were already matched(for features: (1,2)=(2,1) ), don't match again
                std::pair<int, int> inversePair(imgId2, imgId1);
                if(featureMatches.find(inversePair) != featureMatches.end())
                {
                    for(const auto& [featIdx1, featIdx2] : featureMatches[inversePair])
                    {
                        featureMatches[curPair][featIdx2] = featIdx1;
                    }
                    // featureMatches[curPair] = featureMatches[inversePair];
                    continue;
                }

                std::cout << "img1: " << imgIds2Paths[imgId1] 
                          << "| img2: " << imgIds2Paths[imgId2] << std::endl;

                featMatcher->matchFeatures(features1, features2, curMatches, imgIdx2imgShape[imgId1], imgIdx2imgShape[imgId2]);
                

                if(filter)
                {
                    // write matches into vectors
                    std::vector<FeaturePtr<>> featuresMatched1;
                    std::vector<FeaturePtr<>> featuresMatched2;

                    for(const auto& [featIdx1, featIdx2] : curMatches)
                    {
                        featuresMatched1.push_back(features1[featIdx1]);
                        featuresMatched2.push_back(features2[featIdx2]);
                    }

                    // apply geometric filter and leave only valid matches
                    std::shared_ptr<std::vector<bool>> featMatchInliers = std::make_shared<std::vector<bool>>();
                    featFilter->estimateFundamental(featuresMatched1, featuresMatched2, featMatchInliers);

                    // write only filtered matches
                    // std::vector<Match> curMatchesFiltered;
                    std::unordered_map<int, int> curMatchesFiltered;


                    for(size_t matchIdx = 0; matchIdx < featMatchInliers->size(); ++matchIdx)
                    {
                        // if true -> valid match
                        if(featMatchInliers->at(matchIdx))
                        {
                            curMatchesFiltered[matchIdx] = curMatches[matchIdx];
                            // curMatchesFiltered.push_back(curMatches[matchIdx]);
                        }
                    }

                    // std::cout << "matchesInitial: " << curMatches.size() << "| matchesFiltered: " << curMatchesFiltered.size() << std::endl; 
                    // featureMatches[curPair] = curMatchesFiltered;

                    for(const auto& [featIdx1, featIdx2] : curMatchesFiltered)
                    {
                        featureMatches[curPair][featIdx1] = featIdx2;
                    }
                }
                else
                {
                    for(const auto& [featIdx1, featIdx2] : curMatches)
                    {
                        featureMatches[curPair][featIdx1] = featIdx2;
                    }

                    for(const auto& [featIdx1, featIdx2] : curMatches)
                    {
                        
                        // std::cout << "matches: " << featureMatches[curPair][featIdx1] << std::endl;
                        // std::cout << "curMatches: " << curMatches[featIdx1] << std::endl;
                    }

                    // featureMatches[curPair] = curMatches;
                }

                
            }
        }
    }

    void essentialMatToPose(const Eigen::Matrix3d& essentialMat,
                             const std::vector<FeaturePtr<>>& features1,
                             const std::vector<FeaturePtr<>>& features2,
                             const PinholeCamera& intrinsics1,
                             const PinholeCamera& intrinsics2,
                             Eigen::Matrix4d& relativePose)
    {
        auto essentialMatCV = reconstructor::Utils::eigen3dToCVMat(essentialMat, CV_64F);

        auto featuresCV1 = reconstructor::Utils::featuresToCvPoints(features1);
        auto featuresCV2 = reconstructor::Utils::featuresToCvPoints(features2);

        // auto intrinsicsCV1 = reconstructor::Utils::eigen3dToCVMat(intrinsics1);
        // auto intrinsicsCV2 = reconstructor::Utils::eigen3dToCVMat(intrinsics2);

        auto intrinsicsCV1 = intrinsics1.getMatrixCV();
        auto distortCV1 = intrinsics1.getDistortCV();

        auto intrinsicsCV2 = intrinsics2.getMatrixCV();
        auto distortCV2 = intrinsics2.getDistortCV();


        cv::Mat R,t;
        // cv::recoverPose(essentialMatCV, featuresCV1, featuresCV2, intrinsicsCV1, R, t);
        cv::recoverPose(featuresCV1, featuresCV2, intrinsicsCV1, distortCV1, intrinsicsCV2, distortCV2, essentialMatCV, R, t);

        std::cout << "R: " << R << std::endl;
        std::cout << "t: " << t << std::endl;
                
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

    bool sortMatches(std::pair<std::pair<int, int>, std::unordered_map<int, int> > matches1,
                     std::pair<std::pair<int, int>, std::unordered_map<int, int> > matches2)
    {
        return matches1.second.size() > matches2.second.size();
    }

    // 1. find pair with highest number of matches
    Eigen::Matrix4d SequentialReconstructor::chooseInitialPair(int& imgIdx1, int& imgIdx2)
    {
        std::cout << "chooseInitialPair" << std::endl;
        // sort matched images by number of matches:
        std::vector<std::pair<std::pair<int, int>, std::unordered_map<int,int>>> featureMatchesVec;

        for(const auto& match : featureMatches)
        {
            featureMatchesVec.push_back(match);
        }

        // sort by value size
        std::sort(featureMatchesVec.begin(), featureMatchesVec.end(), sortMatches);

        imgIdx1 = featureMatchesVec[0].first.first;
        imgIdx2 = featureMatchesVec[0].first.second;

        // backproject feature coords(K^{-1}*x)
        PinholeCamera intrinsics1(imgIdx2imgShape[imgIdx1].first,
                                  imgIdx2imgShape[imgIdx1].second,
                                  defaultFocalLengthPx,
                                  defaultFocalLengthPx);

        PinholeCamera intrinsics2(imgIdx2imgShape[imgIdx2].first,
                                  imgIdx2imgShape[imgIdx2].second,
                                  defaultFocalLengthPx,
                                  defaultFocalLengthPx);

        imgIdx2camIntrinsics[imgIdx1] = intrinsics1;
        imgIdx2camIntrinsics[imgIdx2] = intrinsics2;

        // extract matches between given images:
        auto imgPair = std::make_pair(imgIdx1, imgIdx2);
        auto featMatches = featureMatches[imgPair];      
        std::vector<FeaturePtr<>> features1;
        std::vector<FeaturePtr<>> features2;

        for(const auto& [featIdx1, featIdx2] : featMatches)
        {
            // std::cout << "featIdx1: " << featIdx1 << "| featIdx2: " << featIdx2 << std::endl;
            // feat
            features1.push_back(features[imgIdx1][featIdx1]);
            features2.push_back(features[imgIdx2][featIdx2]);
            // features2.push_back()
        }
        
        std::shared_ptr<std::vector<bool>> inliers = std::make_shared<std::vector<bool>>();
        auto essentialMat = featFilter->estimateEssential(features1, features2, intrinsics1, intrinsics2, inliers);

        int essentialInliers = 0;
        for(size_t i = 0; i < inliers->size(); ++i)
        {
            if(inliers->at(i))
            {
                ++essentialInliers;
            }
        }
        std::cout << "essentialInliers: " << essentialInliers
                  << "features: " << features1.size() << std::endl;

        // std::cout << "essentialMat: " << essentialMat << std::endl;

        Eigen::Matrix4d relativePose = Eigen::Matrix4d::Identity();
        essentialMatToPose(essentialMat, features1, features2, intrinsics1, intrinsics2, relativePose);

        return relativePose;
    }

    void triangulate2View(const double x1, const double y1,
                          const double x2, const double y2,
                          const Eigen::Matrix<double, 3, 4>& projection1,
                          const Eigen::Matrix<double, 3, 4>& projection2,
                          Eigen::Vector3d& landmark)
    {
        auto row1 = x1 * projection1.row(2) - projection1.row(0);
        auto row2 = y1 * projection1.row(2) - projection1.row(1);
        auto row3 = x2 * projection2.row(2) - projection2.row(0);
        auto row4 = y2 * projection2.row(2) - projection2.row(1);

        Eigen::Matrix4d A;
        A.row(0) = row1;
        A.row(1) = row2;
        A.row(2) = row3;
        A.row(3) = row4;

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        landmark = svd.matrixV().col(3).hnormalized();

        std::cout << "2 view singular value: " << svd.singularValues()(3) << std::endl;
        std::cout << "landmark: " << landmark.transpose() << std::endl; 
    }

    void SequentialReconstructor::triangulateInitialPair(const int imgIdx1,
                                                         const int imgIdx2)
    {
        std::cout << "triangulateInitialPair" << std::endl;
        auto matchedFeatIds = featureMatches[std::make_pair(imgIdx1, imgIdx2)];

        for(const auto& [featIdx1, featIdx2] : matchedFeatIds)
        {
            std::vector<std::pair<int, int>> matchedImgIdFeatId;

            matchedImgIdFeatId.push_back(std::make_pair(imgIdx1,featIdx1));
            matchedImgIdFeatId.push_back(std::make_pair(imgIdx2,featIdx2));


            triangulateMultiView(matchedImgIdFeatId, true);
        }

        std::cout << "landmarks initial size: " << landmarks.size() << std::endl;
    }

    // for those matched with landmark
    // 1. calculate projection error of matched landmark + positive depthness
    // for those not matched with landmark:
    // 1. 2 view triangulation

    void SequentialReconstructor::triangulateCV(const std::vector<std::pair<int, int>> matchedImgIdFeatId)
    {
        auto imgIdx1 = matchedImgIdFeatId[0].first;
        auto featIdx1 = matchedImgIdFeatId[0].second;

        auto imgIdx2 = matchedImgIdFeatId[1].first;
        auto featIdx2 = matchedImgIdFeatId[1].second;

        auto camera1 = imgIdx2camIntrinsics[imgIdx1];
        auto extrinsics1 = imgIdx2camPose[imgIdx1];
        cv::Mat projection1(3, 4, CV_64F);
        for(size_t i = 0; i < 3; ++i)
        {
            for(size_t j = 0; j < 3; ++j)
            {
                projection1.at<double>(i,j) = extrinsics1(i,j);
            }
            projection1.at<double>(i,3) = extrinsics1(i,3);
        }

        auto camera2 = imgIdx2camIntrinsics[imgIdx2];
        auto extrinsics2 = imgIdx2camPose[imgIdx2];
        cv::Mat projection2(3, 4, CV_64F);
        for(size_t i = 0; i < 3; ++i)
        {
            for(size_t j = 0; j < 3; ++j)
            {
                projection2.at<double>(i,j) = extrinsics2(i,j);
            }
            projection2.at<double>(i,3) = extrinsics2(i,3);
        }
        
        cv::Mat pnts3D(1, 2, CV_64FC4);
        cv::Mat cam0pnts(1, 2, CV_64FC2);
        cv::Mat cam1pnts(1, 2, CV_64FC2);


        // cv::triangutePoints();
    }


    void SequentialReconstructor::triangulateMultiView(const std::vector<std::pair<int, int>> matchedImgIdFeatId,
                                                       bool initialCloud)
    {
        Eigen::MatrixXd A;
        A.resize(2 * matchedImgIdFeatId.size(), 4);

        for(size_t pairIdx = 0; pairIdx < matchedImgIdFeatId.size(); ++pairIdx)
        {
            auto imgIdx = matchedImgIdFeatId[pairIdx].first;
            auto featIdx = matchedImgIdFeatId[pairIdx].second;
            
            auto featPtr = features[imgIdx][featIdx];

            auto extrinsics = imgIdx2camPose[imgIdx].block<3,4>(0,0);
            auto intrinsics = imgIdx2camIntrinsics[imgIdx];

            auto projection = extrinsics;

            auto coord3d = intrinsics.unproject(Eigen::Vector2d(featPtr->featCoord.x,
                                                                featPtr->featCoord.y));
            // project on camera plane
            // auto xCoord = (featPtr->featCoord.x - intrinsics(0,2))/intrinsics(0,0);
            // auto yCoord = (featPtr->featCoord.y - intrinsics(1,2))/intrinsics(1,1);

            A.row(2*pairIdx) = coord3d(0) * projection.row(2) - projection.row(0);    
            A.row(2*pairIdx + 1) = coord3d(1) * projection.row(2) - projection.row(1);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        
        if(svd.singularValues()(3))
        {
            auto landmarkCoords = svd.matrixV().col(3).hnormalized();

            if(landmarkCoords(2) < 0 )
            {
                return;
            }

            // std::cout << "svd.singularValues(): " << svd.singularValues() << std::endl;
            // std::cout << "landmarkCoords: " << landmarkCoords << std::endl;

            auto featPtr0 = features[matchedImgIdFeatId[0].first][matchedImgIdFeatId[0].second];
            Landmark landmarkCurr(landmarkCoords(0),landmarkCoords(1), landmarkCoords(2),
                                  featPtr0->featColor.red,
                                  featPtr0->featColor.green,
                                  featPtr0->featColor.blue);

            landmarkCurr.initialLandmark = initialCloud;
             
            for(size_t pairIdx = 0; pairIdx < matchedImgIdFeatId.size(); ++pairIdx)
            {
                auto imgIdx = matchedImgIdFeatId[pairIdx].first;
                auto featIdx = matchedImgIdFeatId[pairIdx].second;

                // auto intrinsics = getIntrinsics(imgIdx2imgShape[imgIdx].first,
                //                                 imgIdx2imgShape[imgIdx].second);

                auto intrinsics = imgIdx2camIntrinsics[imgIdx];

                // check projection error:
                auto landmarkCamFrame = imgIdx2camPose[imgIdx].block<3,3>(0,0) * landmarkCoords + imgIdx2camPose[imgIdx].block<3,1>(0,3);
                
                auto landmarkProjection = intrinsics.project(landmarkCamFrame);

                // auto landmarkProjectionX = intrinsics(0,0) * (landmarkCamFrame(0) / landmarkCamFrame(2)) + intrinsics(0,2);
                // auto landmarkProjectionY = intrinsics(1,1) * (landmarkCamFrame(1) / landmarkCamFrame(2)) + intrinsics(1,2);

                auto residualX = abs(landmarkProjection(0) - features[imgIdx][featIdx]->featCoord.x);
                auto residualY = abs(landmarkProjection(1) - features[imgIdx][featIdx]->featCoord.y);

                auto residualTotal = residualX + residualY; 
                
                if(residualTotal > 30)
                {
                    return;
                }

                std::cout << "imgIdx: " << imgIdx << "| featIdx: " << featIdx 
                          << "| residual: " << residualX + residualY << std::endl;  

                TriangulatedFeature triangulatedFeature(imgIdx, featIdx);

                landmarkCurr.triangulatedFeatures.push_back(triangulatedFeature);

                auto featPtr = features[imgIdx][featIdx];
                featPtr->landmarkId = landmarks.size();
            }

            landmarks.push_back(landmarkCurr);
        }

    }

    void SequentialReconstructor::triangulateMatchedLandmarks(const int imgIdx,
                                                              const std::vector<int>& featureIds,
                                                              const std::vector<int>& landmarkIds)
    {
        // step1. add landmark reference to newly added features
        for(size_t pairIdx = 0; pairIdx < featureIds.size(); ++pairIdx)
        {
            auto featIdx = featureIds[pairIdx];
            auto landmarkId = landmarkIds[pairIdx];

            // 1. check depth, it should be positive 
            auto cameraPose = imgIdx2camPose[imgIdx];
            auto landmarkWorldFrame = Eigen::Vector3d(landmarks[landmarkId].x,
                                                      landmarks[landmarkId].y,
                                                      landmarks[landmarkId].z);

            auto landmarkCameraFrame = cameraPose.block<3,3>(0,0) * landmarkWorldFrame + cameraPose.block<3,1>(0,3);

            if(landmarkCameraFrame(2) > 0)
            {
                TriangulatedFeature feat(imgIdx, featIdx);
                landmarks[landmarkId].triangulatedFeatures.push_back(feat);
                features[imgIdx][featIdx]->landmarkId = landmarkId;
            }

            
        }
        
        // step2. new landmarks creation & triangulation
        // find those features that were not matched with landmark, 
        //  if they have any matched feature from other images -> triangulate

        auto imgFeats = features[imgIdx];

        for(size_t featIdx = 0; featIdx < imgFeats.size(); ++featIdx)
        {
            auto featPtr = imgFeats[featIdx];

            
            if(featPtr->landmarkId == -1)
            {
                // if not yet triangulated, try to find matches among previously registered images
                for(const auto& [regImgIdx, regStatus] : registeredImages)
                {
                    auto curImgMatches = imgMatches[imgIdx];
                    // check if img is registerd and matched with cur image
                    if(regStatus && std::find(curImgMatches.begin(), curImgMatches.end(), regImgIdx) != curImgMatches.end())
                    {
                        // check if there is a match for a given featIdx
                        auto imgPairMatches = featureMatches[std::make_pair(imgIdx, regImgIdx)];
                        if(imgPairMatches.count(featIdx) != 0)
                        {
                            auto matchedFeatId  = imgPairMatches[featIdx];

                            std::vector<std::pair<int, int>> matchedImgIdFeatId;
                            matchedImgIdFeatId.push_back(std::make_pair(regImgIdx, matchedFeatId));
                            matchedImgIdFeatId.push_back(std::make_pair(imgIdx, featIdx));

                            triangulateMultiView(matchedImgIdFeatId);            
                            break;
                        }
                    }
                }
            }
        
        }

        std::cout << "landmarks.size(): " << landmarks.size() << std::endl;

    }

    Eigen::Matrix4d SequentialReconstructor::registerImagePnP(const int imgIdx,
                                                              const std::vector<int>& featureIdxs,
                                                              const std::vector<int>& landmarkIdxs)
    {
        std::cout << "registerImagePnP" << std::endl;

        // convert inputs to opencv compatible format
        cv::Mat landmarksCV(landmarkIdxs.size(), 3, CV_64F);
        cv::Mat featuresCV(featureIdxs.size(), 2, CV_64F);

        for(size_t pairIdx = 0; pairIdx <  featureIdxs.size(); ++pairIdx)
        {
            int featIdx = featureIdxs[pairIdx];
            int landmarkIdx = landmarkIdxs[pairIdx];

            featuresCV.at<double>(pairIdx, 0) = features[imgIdx][featIdx]->featCoord.x;
            featuresCV.at<double>(pairIdx, 1) = features[imgIdx][featIdx]->featCoord.y;

            landmarksCV.at<double>(pairIdx, 0) = landmarks[landmarkIdx].x;
            landmarksCV.at<double>(pairIdx, 1) = landmarks[landmarkIdx].y;
            landmarksCV.at<double>(pairIdx, 2) = landmarks[landmarkIdx].z;
        }

        // 
        
        // auto intrinsics = getIntrinsics(imgIdx2imgShape[imgIdx].first,
        //                                  imgIdx2imgShape[imgIdx].second);

        auto intrinsics = imgIdx2camIntrinsics[imgIdx];                                         

        auto intrinsicsCV = intrinsics.getMatrixCV();
        auto distCoeffsCV = intrinsics.getDistortCV();
        
        // cv::Mat intrinsicsCV = reconstructor::Utils::eigen3dToCVMat(intrinsics);


        cv::Mat rotationCV, translationCV;
        cv::Mat inliersCV;

        cv::solvePnPRansac(landmarksCV,
                           featuresCV,
                           intrinsicsCV,
                           distCoeffsCV,
                           rotationCV, translationCV,
                           false, 10000, 8.0, 0.99,
                           inliersCV);

        std::vector<bool> inliers;    
        reconstructor::Utils::writeInliersToVector(inliersCV, inliers);

        int numInliers = 0;
        for(size_t pairIdx = 0; pairIdx < inliers.size(); ++pairIdx)
        {
            if(inliers[pairIdx])
            {
                ++numInliers;
            }
        }

        Eigen::Matrix4d cameraPose = Eigen::Matrix4d::Identity();

        cv::Mat rotationMatCV;
        cv::Rodrigues(rotationCV, rotationMatCV);

        for(size_t row = 0; row < 3; ++row)
        {
            for(size_t col = 0; col < 3; ++col)
            {
                cameraPose(row, col) = rotationMatCV.at<double>(row,col);
            }
            cameraPose(row, 3) = translationCV.at<double>(row);
        }

        std::cout << "imgIdx: " << imgIdx 
                  << "numInliers: " << numInliers
                  << "totalMatches: " << inliers.size()
                  << "rotation: " << rotationCV
                  << "translation: " << translationCV
                  << std::endl;
        return cameraPose;
    }

    void SequentialReconstructor::addNextView()
    {
        // 1. extract matched images ids
        std::set<int> nextViewCandidates;

        for(const auto& [imgId, imgPath] : imgIds2Paths)
        {
            // if not registered -> this is a candidate for registration
            if(registeredImages.find(imgId) == registeredImages.end())
            {
                nextViewCandidates.insert(imgId);
            }
        }
        std::cout << "n views sequential reconstruction: " << nextViewCandidates.size() << std::endl;

        // sort possible candidates by number of matches with landmarks 
        std::map<int, int> landmarkMatches2ImageIdx; 
        // stores landmark indices per imgIdx { imgIdx : landmarkIdx } 
        std::unordered_map<int, std::vector<int>> imgIdToLandmarkIds;
        // stores feature corresponding to landmark {imgIdx : featureIdx}
        std::unordered_map<int, std::vector<int>> imgIdToFeatureIds;

        for(const auto& candidateImgIdx : nextViewCandidates)
        {
            // get number features, which were triangulated(became landmark)
            int nTriangulatedFeats = 0;
            auto candidateImgMatches = imgMatches[candidateImgIdx];

            std::vector<int> landmarkIds;
            std::vector<int> featureIds;

            for(size_t landmarkIdx = 0; landmarkIdx < landmarks.size(); ++landmarkIdx)
            {
                // check all features matched with this landmark:
                for(const auto& landmarkFeat : landmarks[landmarkIdx].triangulatedFeatures)
                {
                    auto imgIdx = landmarkFeat.imgIdx;
                    auto featIdx = landmarkFeat.featIdx;

                    // check only if candidate img has matches with landmark's imgId
                    if(std::find(candidateImgMatches.begin(), candidateImgMatches.end(), imgIdx) != candidateImgMatches.end())
                    {
                        // get 
                        auto featMatches = featureMatches[std::make_pair(imgIdx, candidateImgIdx)];

                        // find feature in candidateImg which is matched with landmark
                        auto matchedFeatIdx = featMatches.find(featIdx);

                        if(matchedFeatIdx != featMatches.end())
                        {
                            ++nTriangulatedFeats;

                            landmarkIds.push_back(landmarkIdx);
                            featureIds.push_back(matchedFeatIdx->second);
                            // break;
                        }
                    }
                }
            }

            imgIdToLandmarkIds[candidateImgIdx] = landmarkIds;
            imgIdToFeatureIds[candidateImgIdx] = featureIds;

            landmarkMatches2ImageIdx[nTriangulatedFeats] = candidateImgIdx;
        }
        
        // choose image with highest number of matches to landmarks
        auto registeredImgIdx = landmarkMatches2ImageIdx.rbegin()->second;

        PinholeCamera cameraIntrinsics(imgIdx2imgShape[registeredImgIdx].first,
                                       imgIdx2imgShape[registeredImgIdx].second,
                                       defaultFocalLengthPx,
                                       defaultFocalLengthPx);
        imgIdx2camIntrinsics[registeredImgIdx] = cameraIntrinsics;

        // need [landmark - keypoint] matches for PnP
        auto cameraPose = registerImagePnP(registeredImgIdx,
                                           imgIdToFeatureIds[registeredImgIdx],
                                           imgIdToLandmarkIds[registeredImgIdx]);

        imgIdx2camPose[registeredImgIdx] = cameraPose;

        // need [keypoint - keypoint] both with poses for triangulation
        // find all matches for currently registered
        triangulateMatchedLandmarks(registeredImgIdx,
                                    imgIdToFeatureIds[registeredImgIdx],
                                    imgIdToLandmarkIds[registeredImgIdx]);
        
        registeredImages[registeredImgIdx] = true;
        imgIdxOrder.push_back(registeredImgIdx);
        std::cout << "registering imgIdx: " << registeredImgIdx << std::endl;
        std::cout << "landmarks.size(): " << landmarks.size() << std::endl;
    }

    void SequentialReconstructor::reconstruct(const std::string& imgFolder,
                                              const std::string& outFolder)
    {
        // extract all images from a given path and assign them ids
        int imgId = 0;
        for(const auto& entry : fs::directory_iterator(imgFolder))
        {
            // auto imgId2PathPair = std::make_pair(imgId, entry.path());
            // imgIds2Paths.push_back(imgId2PathPair);
            imgIds2Paths[imgId] = entry.path();
            ++imgId;
        }
        // cameraPoses.resize(imgIds2Paths.size());
        // registeredImages.resize(imgIds2Paths.size());
        // std::fill(registeredImages.begin(), registeredImages.end(), false);

        detectFeatures();

        matchImages();

        matchFeatures(false);
        drawFeatMatchesAndSave(outFolder);

        int imgIdx1, imgIdx2;
        auto initialPose = chooseInitialPair(imgIdx1, imgIdx2);


        imgIdx2camPose[imgIdx1] = Eigen::Matrix4d::Identity();
        imgIdx2camPose[imgIdx2] = initialPose;
        registeredImages[imgIdx1] = true;
        registeredImages[imgIdx2] = true;
        imgIdxOrder.push_back(imgIdx1);
        imgIdxOrder.push_back(imgIdx2);


        std::cout << "initial pair imgIdx1: " << imgIdx1 
                               << "imgIdx2: " << imgIdx2 << std::endl; 

        std::cout << "imgIdx2camPose[imgIdx1]: " << imgIdx2camPose[imgIdx1] << std::endl;

        std::cout << "imgIdx2camPose[imgIdx2]: " << imgIdx2camPose[imgIdx2] << std::endl;

        triangulateInitialPair(imgIdx1, imgIdx2);
        std::cout << "landmarks initial size: " << landmarks.size() << std::endl;

        std::vector<Eigen::Vector3d> landmarksUpdated;
        std::vector<Eigen::Vector3d> cameraPosesUpdated;
        // add rest of views via solvepnp
        for(size_t i = 0; i < imgIds2Paths.size()-2; ++i)
        {

            // save initial cloud with camera poses
            std::string cloudPath = "../out_data/clouds/cloud_" + std::to_string(i) + ".ply";
            reconstructor::Utils::saveCloud(landmarks, imgIdx2camPose, cloudPath);

            landmarksUpdated.clear();
            cameraPosesUpdated.clear();
            addNextView();

            BundleAdjuster bundleAdjuster;
            auto camGlobal2LocalIdx = bundleAdjuster.adjust(features,
                                                            landmarks,
                                                            imgIdx2camPose,
                                                            imgIdx2camIntrinsics,
                                                            imgIdxOrder);
            
        }


        std::string cloudPath = "../out_data/clouds/cloud_final.ply";
        reconstructor::Utils::saveCloud(landmarks, imgIdx2camPose, cloudPath);

        auto landmarkCloudPtrAfter = reconstructor::Utils::vectorToPclCloud(landmarksUpdated, 0, 253, 0);
        auto cameraCloudPtrAfter = reconstructor::Utils::vectorToPclCloud(cameraPosesUpdated, 0, 253, 0);

        reconstructor::Utils::viewCloud(landmarkCloudPtrAfter, cameraCloudPtrAfter);
        // reconstructor::Utils::viewCloud(landmarkCloudPtrBefore, landmarkCloudPtrAfter, cameraCloudPtrBefore, cameraCloudPtrAfter);

    }


}
