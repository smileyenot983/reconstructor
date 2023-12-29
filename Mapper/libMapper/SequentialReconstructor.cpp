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

    void SequentialReconstructor::drawFeaturesAndSave(const std::string& outFolder)
    {

        // for(const auto& img)
    }

    void SequentialReconstructor::detectFeatures()
    {
        std::cout << "feature detection" << std::endl;
        for(const auto& [imgId, imgPath] : imgIds2Paths)
        {
            std::cout << "imgId: " << imgId << std::endl;
            auto imgGray = reconstructor::Utils::readGrayImg(imgPath, imgMaxSize);
            auto imgPrepared = featDetector->prepImg(imgGray);

            imgShapes.push_back(std::make_pair(imgPrepared.rows, imgPrepared.cols));

            std::vector<FeaturePtr<>> imgFeatures;
            featDetector->detect(imgPrepared, imgFeatures);

            // features.push_back(imgFeatures);
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

                featMatcher->matchFeatures(features1, features2, curMatches, imgShapes[imgId1], imgShapes[imgId2]);
                

                

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
                    featureMatches[curPair] = curMatchesFiltered;
                }
                else
                {
                    featureMatches[curPair] = curMatches;
                }

                
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

        // std::cout << "featMatches.size(): " << featMatches.size() << std::endl;

        for(const auto& [featIdx1, featIdx2] : featMatches)
        {
            // std::cout << "featIdx1: " << featIdx1 << "| featIdx2: " << featIdx2 << std::endl;
            // feat
            features1.push_back(features[imgIdx1][featIdx1]);
            features2.push_back(features[imgIdx2][featIdx2]);
            // features2.push_back()
        }
        
        auto essentialMat = featFilter->estimateEssential(features1, features2, intrinsics1, intrinsics2);

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

    void SequentialReconstructor::triangulateInitialPair( const int imgIdx1,
                                                         const int imgIdx2)
    {
        std::cout << "triangulateInitialPair" << std::endl;
        auto matchedFeatIds = featureMatches[std::make_pair(imgIdx1, imgIdx2)];

        for(const auto& [featIdx1, featIdx2] : matchedFeatIds)
        {
            std::vector<std::pair<int, int>> matchedImgIdFeatId;

            matchedImgIdFeatId.push_back(std::make_pair(imgIdx1,featIdx1));
            matchedImgIdFeatId.push_back(std::make_pair(imgIdx2,featIdx2));


            triangulateMultiView(matchedImgIdFeatId);



            // auto featPtr1 = features[imgIdx1][featIdx1];
            // auto featPtr2 = features[imgIdx2][featIdx2];

            // auto intrinsics1 = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
            //                                                      imgShapes[imgIdx1].first,
            //                                                      imgShapes[imgIdx1].second,
            //                                                      defaultFov);

            // auto intrinsics2 = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
            //                                                         imgShapes[imgIdx2].first,
            //                                                         imgShapes[imgIdx2].second,
            //                                                         defaultFov);

            // auto x1 = (featPtr1->featCoord.x - intrinsics1(0,2)) / intrinsics1(0,0);
            // auto y1 = (featPtr1->featCoord.y - intrinsics1(1,2)) / intrinsics1(1,1);
            // auto x2 = (featPtr2->featCoord.x - intrinsics2(0,2)) / intrinsics2(0,0);
            // auto y2 = (featPtr2->featCoord.y - intrinsics2(1,2)) / intrinsics2(1,1);

            // auto projection1 = Eigen::Matrix4d::Identity().block<3, 4>(0,0);
            // auto projection2 = relativePose.block<3, 4>(0,0); 

            // auto x1 = featPtr1->featCoord.x;
            // auto y1 = featPtr1->featCoord.y;
            // auto x2 = featPtr2->featCoord.x;
            // auto y2 = featPtr2->featCoord.y;

            // auto projection1 = intrinsics1 * Eigen::Matrix4d::Identity().block<3, 4>(0,0);
            // auto projection2 = intrinsics2 * relativePose.block<3, 4>(0,0); 

            // Eigen::Vector3d landmarkCoords;
            // triangulate2View(x1, y1,
            //                  x2, y2,
            //                  projection1,
            //                  projection2,
            //                  landmarkCoords);

            // Landmark landmarkCurr(landmarkCoords(0),landmarkCoords(1), landmarkCoords(2));
            // TriangulatedFeature triangulatedFeature1(imgIdx1, featIdx1);
            // TriangulatedFeature triangulatedFeature2(imgIdx2, featIdx2);

            // landmarkCurr.triangulatedFeatures.push_back(triangulatedFeature1);
            // landmarkCurr.triangulatedFeatures.push_back(triangulatedFeature2);

            // landmarks.push_back(landmarkCurr);

            // featPtr1->landmarkId = landmarks.size()-1;
            // featPtr2->landmarkId = landmarks.size()-1;
            
        }

        std::cout << "landmarks initial size: " << landmarks.size() << std::endl;
    }

    void SequentialReconstructor::triangulateMultiView(const std::vector<std::pair<int, int>> matchedImgIdFeatId)
    {
        Eigen::MatrixXd A;
        A.resize(2 * matchedImgIdFeatId.size(), 4);

        for(size_t pairIdx = 0; pairIdx < matchedImgIdFeatId.size(); ++pairIdx)
        {
            auto imgIdx = matchedImgIdFeatId[pairIdx].first;
            auto featIdx = matchedImgIdFeatId[pairIdx].second;
            
            auto featPtr = features[imgIdx][featIdx];

            auto intrinsics = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                                 imgShapes[imgIdx].first,
                                                                 imgShapes[imgIdx].second,
                                                                 defaultFov);

            auto extrinsics = imgIdx2camPose[imgIdx].block<3,4>(0,0);

            auto projection = intrinsics * extrinsics;


            A.row(2*pairIdx) = featPtr->featCoord.x * projection.row(2) - projection.row(0);    
            A.row(2*pairIdx + 1) = featPtr->featCoord.y * projection.row(2) - projection.row(1);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        
        // if(svd.singularValues()(3) < 100)
        {
            auto landmarkCoords = svd.matrixV().col(3).hnormalized();

            std::cout << "svd.singularValues(): " << svd.singularValues() << std::endl;
            std::cout << "landmarkCoords: " << landmarkCoords << std::endl;

            Landmark landmarkCurr(landmarkCoords(0),landmarkCoords(1), landmarkCoords(2));

            for(size_t pairIdx = 0; pairIdx < matchedImgIdFeatId.size(); ++pairIdx)
            {
                auto imgIdx = matchedImgIdFeatId[pairIdx].first;
                auto featIdx = matchedImgIdFeatId[pairIdx].second;

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

            TriangulatedFeature feat(imgIdx, featIdx);
            landmarks[landmarkId].triangulatedFeatures.push_back(feat);
            features[imgIdx][featIdx]->landmarkId = landmarkId;
        }
        
        // step2. new landmarks creation & triangulation
        // find those features that were not matched with landmark, 
        //  if they have any matched feature from other images -> triangulate

        auto imgFeats = features[imgIdx];

        for(size_t featIdx = 0; featIdx < imgFeats.size(); ++featIdx)
        {
            auto featPtr = imgFeats[featIdx];

            std::vector<std::pair<int, int>> matchedImgIdFeatId;
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

                            matchedImgIdFeatId.push_back(std::make_pair(regImgIdx, matchedFeatId));
                        }
                    }
                }
            }
            
            if(matchedImgIdFeatId.size() > 0)
            {
                matchedImgIdFeatId.push_back(std::make_pair(imgIdx, featIdx));

                std::cout << "matchedImgIdFeatId.size(): " << matchedImgIdFeatId.size() << std::endl;
                triangulateMultiView(matchedImgIdFeatId);
            }
        }

        std::cout << "landmarks.size(): " << landmarks.size() << std::endl;

            


    }


    void SequentialReconstructor::registerImagePnP(const int imgIdx,
                                                   const std::vector<int>& featureIdxs,
                                                   const std::vector<int>& landmarkIdxs)
    {
        std::cout << "registerImagePnP" << std::endl;
        for(size_t i = 0; i < featureIdxs.size(); ++i)
        {
            auto xCoord = features[imgIdx][featureIdxs[i]]->featCoord.x;
            auto yCoord = features[imgIdx][featureIdxs[i]]->featCoord.y;
        }


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

            // std::cout << "featuresCV.at<double>(4, 0): " << featuresCV.at<double>(4, 0) << std::endl;

        }

        // 
        
        auto intrinsics = reconstructor::Utils::getIntrinsicsMat(defaultFocalLengthmm,
                                                                 imgShapes[imgIdx].first,
                                                                 imgShapes[imgIdx].second,
                                                                 defaultFov);

        cv::Mat intrinsicsCV = reconstructor::Utils::eigen3dToCVMat(intrinsics);


        // cv::Mat distCoeffsCV = cv::Mat::zeros(4, 1, CV_64F);
        cv::Mat rotationCV, translationCV;
        cv::Mat inliersCV;

        // std::cout << "intrinsicsCV: " << intrinsicsCV << std::endl;
        // std::cout << "landmarksCV.size: " << landmarksCV.size << std::endl;
        // std::cout << "featuresCV.size: " << featuresCV.size << std::endl;

        // failed imgIndices : 7, 10, 5 order: 1 8 6 4(fail) 7(fail) 10 2 9(fail) 3 0 5
        cv::solvePnPRansac(landmarksCV,
                           featuresCV,
                           intrinsicsCV,
                           cv::Mat(),
                           rotationCV, translationCV,
                           false, 100, 8.0, 0.99,
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


        std::cout << "imgIdx: " << imgIdx 
                  << "numInliers: " << numInliers
                  << "totalMatches: " << inliers.size()
                  << "rotation: " << rotationCV
                  << "translation: " << translationCV
                  << std::endl;
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
                // for()
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


                            break;
                            // imgIdxToLandmarkIdxs[candidateImgIdx].push_back(landmarkIdx);
                            // imgIdxToFeatureIdxs[candidateImgIdx].push_back(matchedFeatIdx->second);


                        }
                    }
                }
            }

            imgIdToLandmarkIds[candidateImgIdx] = landmarkIds;
            imgIdToFeatureIds[candidateImgIdx] = featureIds;

            landmarkMatches2ImageIdx[nTriangulatedFeats] = candidateImgIdx;

        }


        // colmap: 1. find all existing correspondences for each feature
        //         2. those that are matched to triangulated feature - calc angle between rays in camera cs: ray1 = K^{-1}*feat coord(uv1)
        //                                                                                                   ray2 = extr * landmark coord(XYZ_world)
                                    // ray1^T * ray2 = ||ray1|| * ||ray2|| * cos(ray1, ray2) => cos(ray1, ray2) = ray1^T * ray2 / ||ray1||*||ray2||
        
        // choose image with highest number of matches to landmarks
        auto registeredImgIdx = landmarkMatches2ImageIdx.rbegin()->second;
        
        // need [landmark - keypoint] matches for PnP
        registerImagePnP(registeredImgIdx,
                         imgIdToFeatureIds[registeredImgIdx],
                         imgIdToLandmarkIds[registeredImgIdx]);

        // need [keypoint - keypoint] both with poses for triangulation
        // find all matches for currently registered
        // triangulateMatchedLandmarks(registeredImgIdx,
        //                             imgIdToFeatureIds[registeredImgIdx],
        //                             imgIdToLandmarkIds[registeredImgIdx]);

        

            
        
        registeredImages[registeredImgIdx] = true;
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
        drawFeaturesAndSave(outFolder);

        matchImages();

        matchFeatures(false);
        drawFeatMatchesAndSave(outFolder);

        int imgIdx1, imgIdx2;
        auto initialPose = chooseInitialPair(imgIdx1, imgIdx2);

        imgIdx2camPose[imgIdx1] = Eigen::Matrix4d::Identity();
        imgIdx2camPose[imgIdx2] = initialPose;
        registeredImages[imgIdx1] = true;
        registeredImages[imgIdx2] = true;

        std::cout << "initial pair imgIdx1: " << imgIdx1 
                               << "imgIdx2: " << imgIdx2 << std::endl; 

        triangulateInitialPair(imgIdx1, imgIdx2);

        
        std::cout << "landmarks initial size: " << landmarks.size() << std::endl;

        // add rest of views via solvepnp
        // for(size_t i = 0; i < imgIds2Paths.size()-2; ++i)
        for(size_t i = 0; i < 1; ++i)
        {
            addNextView();
        }

        auto landmarkCloudPtr = reconstructor::Utils::landmarksToPclCloud(landmarks);
        auto cameraCloudPtr = reconstructor::Utils::cameraPosesToPclCloud(imgIdx2camPose);
        reconstructor::Utils::viewCloud(landmarkCloudPtr, cameraCloudPtr);

    }


}
