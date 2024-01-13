#include "SequentialReconstructor.h"

#include <filesystem>

#include <omp.h>

#include "utils.h"
#include "BasicFlags.h"


// TODO:
// ADD ANGLE CHECK

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
        case FeatDetectorType::Classic:
            featDetector = std::make_unique<FeatureClassic>();
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
        // #pragma omp parallel for
        for(const auto& [imgId, imgPath] : imgIds2Paths)
        {
            cv::Mat imgOriginal;
            downScaleFactor = reconstructor::Utils::readImg(imgOriginal, imgPath, imgMaxSize);

            PinholeCamera intrinsics(imgOriginal.rows,
                                     imgOriginal.cols,
                                     defaultFocalLengthFactor * downScaleFactor);
            imgIdx2camIntrinsics[imgId] = intrinsics;

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

            features[imgId] = imgFeatures;
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

        // #pragma omp parallel for
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
                    if(imgType == CV_8U)
                    {
                        imgMerged.at<uchar>(row,col) = imgPrepared1.at<uchar>(row,col);
                    }
                    else if(imgType == CV_32F)
                    {
                        imgMerged.at<float>(row,col) = 255 * imgPrepared1.at<float>(row,col);
                    }
                    else
                    {
                        throw std::runtime_error("Unknown image type!");
                    }
                }
            }
            // now from second:
            int colOffset = imgPrepared1.cols;
            for(size_t row = 0; row < std::min(imgMerged.rows, imgPrepared2.rows); ++row)
            {
                for(size_t col = 0; col < imgPrepared2.cols; ++col)
                {
                    if(imgType == CV_8U)
                    {
                        imgMerged.at<uchar>(row,col + colOffset) = imgPrepared2.at<uchar>(row,col);
                    }
                    else if(imgType == CV_32F)
                    {
                        imgMerged.at<float>(row,col + colOffset) = 255 * imgPrepared2.at<float>(row,col);
                    }
                    else
                    {
                        throw std::runtime_error("Unknown image type!");
                    }
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
        // #pragma omp parallel for
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
                std::map<int, int> curMatches;

                // if this pair's feature were already matched(for features: (1,2)=(2,1) ), don't match again
                std::pair<int, int> inversePair(imgId2, imgId1);
                if(featureMatches.find(inversePair) != featureMatches.end())
                {
                    for(const auto& [featIdx1, featIdx2] : featureMatches[inversePair])
                    {
                        featureMatches[curPair][featIdx2] = featIdx1;
                    }
                    continue;
                }

                std::cout << "img1: " << imgIds2Paths[imgId1] 
                          << "| img2: " << imgIds2Paths[imgId2] << std::endl;

                featMatcher->matchFeatures(features1, features2, curMatches, imgIdx2imgShape[imgId1], imgIdx2imgShape[imgId2]);
                
                std::cout << "curMatches.size(): " << curMatches.size() << std::endl;

                // need at least 7 matched points for fundamental matrix estimation(7 pt algorithm)
                if(filter && curMatches.size() >= 7)
                {
                    // write matches into vectors
                    std::vector<FeaturePtr<>> featuresMatched1;
                    std::vector<FeaturePtr<>> featuresMatched2;
                    
                    for(const auto& [featIdx1, featIdx2] : curMatches)
                    {
                        featuresMatched1.push_back(features1[featIdx1]);
                        featuresMatched2.push_back(features2[featIdx2]);
                    }

                    std::vector<bool> inlierMatchIds;
                    // apply geometric filter and leave only valid matches
                    featFilter->estimateFundamental(featuresMatched1, featuresMatched2, inlierMatchIds);

                    if(inlierMatchIds.size() == 0)
                    {
                        continue;
                    }

                    // write only filtered matches
                    int curMatchId = 0;
                    for(const auto& [featIdx1, featIdx2] : curMatches)
                    {
                        if(inlierMatchIds[curMatchId])
                        {
                            featureMatches[curPair][featIdx1] = featIdx2;
                        }
                        ++curMatchId;
                    }
                    std::cout << "curMatchesFiltered.size(): " << featureMatches[curPair].size() << std::endl;

                }
                else
                {
                    for(const auto& [featIdx1, featIdx2] : curMatches)
                    {
                        featureMatches[curPair][featIdx1] = featIdx2;
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
        // PinholeCamera intrinsics1(imgIdx2imgShape[imgIdx1].first,
        //                           imgIdx2imgShape[imgIdx1].second,
        //                           defaultFocalLengthFactor * downScaleFactor);

        // PinholeCamera intrinsics2(imgIdx2imgShape[imgIdx2].first,
        //                           imgIdx2imgShape[imgIdx2].second,
        //                           defaultFocalLengthFactor * downScaleFactor);

        // imgIdx2camIntrinsics[imgIdx1] = intrinsics1;
        // imgIdx2camIntrinsics[imgIdx2] = intrinsics2;

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
        

        std::vector<bool> inlierMatchIds;
        auto essentialMat = featFilter->estimateEssential(features1, features2, 
                                                          imgIdx2camIntrinsics[imgIdx1],
                                                          imgIdx2camIntrinsics[imgIdx2],
                                                          inlierMatchIds);

        int essentialInliers = 0;
        for(size_t i = 0; i < inlierMatchIds.size(); ++i)
        {
            if(inlierMatchIds[i])
            {
                ++essentialInliers;
            }
        }
        std::cout << "essentialInliers: " << essentialInliers
                  << "features: " << features1.size() << std::endl;

        // std::cout << "essentialMat: " << essentialMat << std::endl;

        Eigen::Matrix4d relativePose = Eigen::Matrix4d::Identity();
        essentialMatToPose(essentialMat, features1, features2, imgIdx2camIntrinsics[imgIdx1], imgIdx2camIntrinsics[imgIdx2], relativePose);

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

    void SequentialReconstructor::triangulateMultiView(const std::vector<std::pair<int, int>> matchedImgIdFeatId,
                                                       bool initialCloud)
    {
        // 1. construct linear system and find nullspace
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

            // project on camera plane
            auto coord3d = intrinsics.unproject(Eigen::Vector2d(featPtr->featCoord.x,
                                                                featPtr->featCoord.y));
            

            A.row(2*pairIdx) = coord3d(0) * projection.row(2) - projection.row(0);    
            A.row(2*pairIdx + 1) = coord3d(1) * projection.row(2) - projection.row(1);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        auto landmarkCoords = svd.matrixV().col(3).hnormalized();

        // check that there are no nans and positive depth
        if(svd.singularValues()(3) && landmarkCoords(2) > 0 )
        {
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

                auto landmarkLocalCoords = getLandmarkLocalCoords(imgIdx, landmarkCurr);
                auto residualTotal = calcProjectionError(imgIdx, featIdx, landmarkLocalCoords);

                if(residualTotal > maxProjectionError)
                {
                    return;
                }
  
                TriangulatedFeature triangulatedFeature(imgIdx, featIdx);
                landmarkCurr.triangulatedFeatures.push_back(triangulatedFeature);
            }

            // check angle between rays c1->X, c2->X:
            for(size_t i = 0; i < landmarkCurr.triangulatedFeatures.size(); ++i)
            {
                auto imgIdx1 = landmarkCurr.triangulatedFeatures[i].imgIdx;
                auto featIdx1 = landmarkCurr.triangulatedFeatures[i].featIdx;

                for(size_t j = 0; j < landmarkCurr.triangulatedFeatures.size(); ++j)
                {
                    if(i != j)
                    {
                        auto imgIdx2 = landmarkCurr.triangulatedFeatures[j].imgIdx;
                        auto featIdx2 = landmarkCurr.triangulatedFeatures[j].featIdx; 
                        
                        auto angle = calcTriangulationAngle(imgIdx1, imgIdx2,
                                                            featIdx1, featIdx2,
                                                            landmarkCurr);

                        if(angle < minTriangulationAngle)
                        {
                            return;
                        }  
                    }

                }
            }

            for(size_t pairIdx = 0; pairIdx < matchedImgIdFeatId.size(); ++pairIdx)
            {
                auto imgIdx = matchedImgIdFeatId[pairIdx].first;
                auto featIdx = matchedImgIdFeatId[pairIdx].second;

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

            auto landmarkLocalCoords = getLandmarkLocalCoords(imgIdx, landmarks[landmarkId]);
            auto residualTotal = calcProjectionError(imgIdx, featIdx, landmarkLocalCoords);

        
            // check that unprojected feature has positive depth and low reprojection error and was 
            if(landmarkLocalCoords(2) > 0 && residualTotal < maxProjectionError && features[imgIdx][featIdx]->landmarkId == -1)
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

                            auto featPtr2 = features[regImgIdx][matchedFeatId];

                            if(featPtr2->landmarkId == -1)
                            {
                                triangulateMultiView(matchedImgIdFeatId);            
                                break;
                            }

                            
                        }
                    }
                }
            }
        
        }

        std::cout << "landmarks.size(): " << landmarks.size() << std::endl;

    }

    Eigen::Matrix4d SequentialReconstructor::registerImagePnP(const int imgIdx,
                                                              std::vector<int>& featureIdxs,
                                                              std::vector<int>& landmarkIdxs)
    {
        int totalMatchSize = featureIdxs.size();
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


        auto intrinsics = imgIdx2camIntrinsics[imgIdx];                                         

        auto intrinsicsCV = intrinsics.getMatrixCV();
        auto distCoeffsCV = intrinsics.getDistortCV();
        
        cv::Mat rotationCV, translationCV;
        cv::Mat inliersCV;

        cv::solvePnPRansac(landmarksCV,
                           featuresCV,
                           intrinsicsCV,
                           distCoeffsCV,
                           rotationCV, translationCV,
                           false, 10000, maxProjectionError, 0.99,
                           inliersCV);

        std::vector<bool> inlierIds;    
        reconstructor::Utils::writeInliersToVector(inliersCV, inlierIds);

        // featureIdxs
        // landmarkIdxs
        std::vector<int> featureIdsInliers;
        std::vector<int> landmarkIdsInliers;
        
        for(size_t pairId = 0; pairId < inlierIds.size(); ++pairId)
        {
            if(inlierIds[pairId])
            {
                featureIdsInliers.push_back(featureIdxs[pairId]);
                landmarkIdsInliers.push_back(landmarkIdxs[pairId]);
            }
        }

        featureIdxs = featureIdsInliers;
        landmarkIdxs = landmarkIdsInliers;



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
                  << "numInliers: " << featureIdxs.size()
                  << "totalMatches: " << totalMatchSize
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

                        if(matchedFeatIdx != featMatches.end() && features[candidateImgIdx][matchedFeatIdx->second]->landmarkId == -1)
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

        // PinholeCamera cameraIntrinsics(imgIdx2imgShape[registeredImgIdx].first,
        //                                imgIdx2imgShape[registeredImgIdx].second,
        //                                defaultFocalLengthFactor * downScaleFactor);
        // imgIdx2camIntrinsics[registeredImgIdx] = cameraIntrinsics;

        std::vector<bool> inlierMatches;
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

    double SequentialReconstructor::calcTriangulationAngle(int imgIdx1, int imgIdx2,
                                                           int featIdx1, int featIdx2,
                                                           const Landmark& landmark)
    {
        auto imgPose1 = imgIdx2camPose[imgIdx1];
        auto imgCenter1 = -imgPose1.block<3,3>(0,0).transpose() * imgPose1.block<3,1>(0,3);
        auto ray1 = Eigen::Vector3d(landmark.x - imgCenter1(0),
                                    landmark.y - imgCenter1(1),
                                    landmark.z - imgCenter1(2));

        auto imgPose2 = imgIdx2camPose[imgIdx2];
        auto imgCenter2 = -imgPose2.block<3,3>(0,0).transpose() * imgPose2.block<3,1>(0,3);
        auto ray2 = Eigen::Vector3d(landmark.x - imgCenter2(0),
                                    landmark.y - imgCenter2(1),
                                    landmark.z - imgCenter2(2));

        auto angleRadians = acos(ray1.dot(ray2) / (ray1.norm() * ray2.norm()));

        auto angleDegrees = 180.0 * angleRadians / 3.1415;

        return angleDegrees;
    }

    // transforms landmark into camera frame and returns coords
    Eigen::Vector3d SequentialReconstructor::getLandmarkLocalCoords(int imgIdx,
                                                                    Landmark landmark)
    {
        auto pose = imgIdx2camPose[imgIdx];
        Eigen::Vector3d landmarkCoordsWorld(landmark.x,
                                                    landmark.y,
                                                    landmark.z);

        auto landmarkCoordsLocal = pose.block<3,3>(0,0) * landmarkCoordsWorld + pose.block<3,1>(0,3);

        return landmarkCoordsLocal;
    }

    double SequentialReconstructor::calcProjectionError(int imgIdx, int featIdx,
                                                        Eigen::Vector3d landmarkCoordsLocal)
    {

        auto feat = features[imgIdx][featIdx];
        auto intrinsics = imgIdx2camIntrinsics[imgIdx];

        // project on image plane
        auto landmarkProjection = intrinsics.project(landmarkCoordsLocal);

        auto residualX = abs(landmarkProjection(0) - feat->featCoord.x);
        auto residualY = abs(landmarkProjection(1) - feat->featCoord.y);
        auto residualTotal = residualX + residualY;

        return residualTotal;
    }

    std::vector<bool> SequentialReconstructor::checkLandmarkValidity()
    {
        // count inliers after BA for all landmarks:
        int numInliers = 0;
        std::vector<bool> inlierLandmarks;
        for(auto& landmark : landmarks)
        {
            bool inlierLandmark = true;

            for(int landFeatId = 0; landFeatId < landmark.triangulatedFeatures.size(); ++landFeatId)
            {
                auto imgId = landmark.triangulatedFeatures[landFeatId].imgIdx;
                auto featId = landmark.triangulatedFeatures[landFeatId].featIdx;

                auto landmarkCoordsLocal = getLandmarkLocalCoords(imgId, landmark);
                auto residualTotal = calcProjectionError(imgId, featId, landmarkCoordsLocal);

                if(residualTotal > maxProjectionError || 
                   landmarkCoordsLocal(2) < 0)
                {
                    std::cout << "outlier triang feat: "
                                << "| imgId: " << imgId
                                << "| featId: " << featId
                                << "| landmarkId: " << features[imgId][featId]->landmarkId
                                << "| residual: " << residualTotal 
                                << "| depth: " << landmarkCoordsLocal(2) << std::endl;

                    landmark.triangulatedFeatures.erase(landmark.triangulatedFeatures.begin() + landFeatId);
                    if(landmark.triangulatedFeatures.size() < 2)
                    {
                        inlierLandmark = false;
                    }
                }
            }

            bool angleCheckPassed = false;
            // calculate angle between all combinations of angles
            for(size_t featId1 = 0; featId1 < landmark.triangulatedFeatures.size(); ++featId1)
            {
                for(size_t featId2 = 0; featId2 < landmark.triangulatedFeatures.size(); ++featId2)
                {
                    if(featId1 == featId2)
                    {
                        continue;
                    }

                    auto angle = calcTriangulationAngle(landmark.triangulatedFeatures[featId1].imgIdx,
                                                        landmark.triangulatedFeatures[featId2].imgIdx,
                                                        landmark.triangulatedFeatures[featId1].featIdx,
                                                        landmark.triangulatedFeatures[featId2].featIdx,
                                                        landmark);

                    if(angle > minTriangulationAngle)
                    {
                        angleCheckPassed = true;
                    }
                    else
                    {
                        auto featIdx1 = landmark.triangulatedFeatures[featId1].featIdx;
                        auto imgIdx1 = landmark.triangulatedFeatures[featId1].imgIdx;
                        auto featIdx2 = landmark.triangulatedFeatures[featId2].featIdx;
                        auto imgIdx2 = landmark.triangulatedFeatures[featId2].imgIdx;
                        std::cout << "low triangulation angle: " << angle
                                  << "| landmarkId1: " << features[imgIdx1][featIdx1]->landmarkId
                                  << "| landmarkId2: " << features[imgIdx2][featIdx2]->landmarkId 
                                  << "| featId1: " << featId1
                                  << "| featId2: " << featId2
                                  << "| imgIdx1: " << landmark.triangulatedFeatures[featId1].imgIdx
                                  << "| imgIdx2: " << landmark.triangulatedFeatures[featId2].imgIdx
                                  << "| featIdx1: " << landmark.triangulatedFeatures[featId1].featIdx
                                  << "| featIdx2: " << landmark.triangulatedFeatures[featId2].featIdx << std::endl;
                    }
                }
            }
            // in case there is at least 1 combination of features producing high triangulation angle
            //    then keep this landmark, otherwise remove
            if(!angleCheckPassed)
            {
                inlierLandmark = false;
            }

            inlierLandmarks.push_back(inlierLandmark);
        }

        return inlierLandmarks;
    }

    void SequentialReconstructor::removeOutlierLandmarks(const std::vector<bool>& inlierIds)
    {
        std::vector<Landmark> updatedLandmarks;
        for(size_t i = 0; i < inlierIds.size(); ++i)
        {
            if(inlierIds[i])
            {
                updatedLandmarks.push_back(landmarks[i]);
                
            }
            else
            {
                for(const auto& triangFeat : landmarks[i].triangulatedFeatures)
                {
                    auto imgIdx = triangFeat.imgIdx;
                    auto featIdx = triangFeat.featIdx;
                    features[imgIdx][featIdx]->landmarkId = -1;
                }
            }
        }
        landmarks = updatedLandmarks;
    }


    void SequentialReconstructor::reconstruct(const std::string& imgFolder,
                                              const std::string& outFolder)
    {
        reconstructor::Utils::deleteDirectoryContents("../out_data/clouds/");
        reconstructor::Utils::deleteDirectoryContents("../out_data/matches");

        // extract all images from a given path and assign them ids
        int imgId = 0;
        for(const auto& entry : fs::directory_iterator(imgFolder))
        {
            // auto imgId2PathPair = std::make_pair(imgId, entry.path());
            // imgIds2Paths.push_back(imgId2PathPair);
            imgIds2Paths[imgId] = entry.path();
            std::cout << "imgId: " << imgId
                      << "| path: " << entry.path() << std::endl;
            ++imgId;
        }
        
        timeLogger.startEvent("feature extraction");
        detectFeatures();
        timeLogger.endEvent();

        timeLogger.startEvent("image matching");
        matchImages();
        timeLogger.endEvent();

        timeLogger.startEvent("feature matching");
        matchFeatures(true);
        timeLogger.endEvent();

        timeLogger.startEvent("saving matches");
        drawFeatMatchesAndSave(outFolder);
        timeLogger.endEvent();

        timeLogger.startEvent("initial pair and pose estimation");
        int imgIdx1, imgIdx2;
        auto initialPose = chooseInitialPair(imgIdx1, imgIdx2);
        timeLogger.endEvent();

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

        timeLogger.startEvent("initial pair features triangulation");
        triangulateInitialPair(imgIdx1, imgIdx2);
        timeLogger.endEvent();

        std::cout << "landmarks initial size: " << landmarks.size() << std::endl;

        

        // add rest of views via solvepnp
        for(size_t i = 0; i < imgIds2Paths.size()-2; ++i)
        {
            timeLogger.startEvent("adding new view");
            addNextView();
            timeLogger.endEvent();

            

            timeLogger.startEvent("global bundle adjustment");
            
            // count inliers before BA for all landmarks:
            auto inliersBefore = checkLandmarkValidity();

            int nInliersBefore = 0;
            for(const auto landmarkStatus : inliersBefore)
            {
                if(landmarkStatus)
                {
                    ++nInliersBefore;
                }
            }

            // save cloud before BA
            std::string cloudPath = "../out_data/clouds/cloud_before_" + std::to_string(i) + ".ply";
            reconstructor::Utils::saveCloud(landmarks, imgIdx2camPose, cloudPath, inliersBefore);

            BundleAdjuster bundleAdjuster;
            auto camGlobal2LocalIdx = bundleAdjuster.adjust(features,
                                                            landmarks,
                                                            imgIdx2camPose,
                                                            imgIdx2camIntrinsics,
                                                            imgIdxOrder);

            auto inliersAfter = checkLandmarkValidity();

            int nInliersAfter = 0;
            for(const auto landmarkStatus : inliersBefore)
            {
                if(landmarkStatus)
                {
                    ++nInliersAfter;
                }
            }

            // remove outlier landmarks
            removeOutlierLandmarks(inliersAfter);

            // save cloud after BA
            cloudPath = "../out_data/clouds/cloud_after_" + std::to_string(i) + ".ply";
            reconstructor::Utils::saveCloud(landmarks, imgIdx2camPose, cloudPath, inliersAfter);

            std::cout << "inliers before: " << nInliersBefore 
                      << "inliers after: " << nInliersAfter 
                      << "total landmarks: " << inliersBefore.size() << std::endl;

            


            timeLogger.endEvent();
        }

        std::cout << "reconstruction order: " << std::endl;
        for(size_t i = 0; i < imgIdxOrder.size(); ++i)
        {
            std::cout << "imgIdx: " << imgIdxOrder[i] << std::endl;
        }


        timeLogger.printTimings();

        std::string cloudPath = "../out_data/clouds/cloud_final.ply";
        reconstructor::Utils::saveCloud(landmarks, imgIdx2camPose, cloudPath);

    }

}
