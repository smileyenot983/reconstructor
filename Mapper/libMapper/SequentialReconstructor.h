#pragma once

#include <string>
#include <filesystem>
#include <unordered_map>

#include "FeatureDetector.h"
#include "FeatureSuperPoint.h"
#include "FeatureMatcher.h"
#include "FeatureMatcherSuperglue.h"
#include "ImageMatcher.h"
#include "GeometricFilter.h"
#include "BundleAdjuster.h"
#include "Camera.h"
#include "TimeLogger.h"

#define MAX_NUM_THREADS 4

namespace fs = std::filesystem;

namespace reconstructor::Core
{
    enum FeatDetectorType
    {
        Classic,
        SuperPoint
    };

    enum FeatMatcherType
    {
        Flann,
        SuperGlue
    };

    enum ImgMatcherType
    {
        Fake,
    };

    /*
        2 modes for next image ranking:
            1. MatchTotal - total number of 2d-3d matches
            2. MatchDensity - uniformness of 2d-3d matches density, inspired by colmap  
    */
    enum NextImageRankingMode
    {
        MatchTotal,
        MatchDensity
    };

    struct pair_hash
    {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const
        {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);

            // Simple hash combining technique (could be more sophisticated)
            return h1 ^ h2;
        }
    };

    /*
    class which performs sequential reconstruction:
    1. Feature detection
    2. Image & feature matching
    3. Sequentially adding new images via solvePnP
    4. bundle adjustment
    */
    class SequentialReconstructor
    {
    public:
        SequentialReconstructor(const FeatDetectorType &featDetectorType,
                                const FeatMatcherType &featMatcherType,
                                const ImgMatcherType &imgMatcherType);

        /*
            Detects keypoints and calculates descriptors
        */
        void detectFeatures();

        /*
            Matches images
        */
        void matchImages();

        /*
            Matches features between all images
        */
        void matchFeatures(bool filter = true);

        /*
            Draws matches between matched features and saves resulting images
        */
        void drawFeatMatchesAndSave(const std::string &outFolder);

        // filter matched features via fundamental matrix
        /*
            Filter feature matches using epipolar geometry and fundamental matrix
            For correct matches epipolar constraint should hold:
            x'^T * F * x = 0
        */
        void filterFeatMatches();

        /*
            Chooses initial image pair to start reconstruction,
            based on total number of matched features
        */
        Eigen::Matrix4d chooseInitialPair(int &imgIdx1, int &imgIdx2);

        /*
            Triangulation of initial pair of images
        */
        void triangulateInitialPair(const int imgIdx1,
                                    const int imgIdx2);

        /*
            Performs linear triangulation via DLT using matched features
        */
        void triangulateMultiView(const std::vector<std::pair<int, int>> matchedImgIdFeatId,
                                  bool initialCloud = false);

        /*
           Performs linear triangulation,
            1. adds references to landmarks about newly matched features
            2. creates new landmarks in case there is a match with feature, which has camera pose,
               but was not triangulated previously
        */
        void triangulateMatchedLandmarks(const int imgIdx,
                                         const std::vector<int> &featureIdxs,
                                         const std::vector<int> &landmarkIdxs);

        /*
            matches features on given images with existing landmarks 
        */
        void calc2d3dMatches(const std::set<int>& candidateImgIds,
                            std::unordered_map<int, std::vector<int>>& imgIdToLandmarkIds,
                            std::unordered_map<int, std::vector<int>>& imgIdToFeatureIds);

        void rankNextImages(const std::unordered_map<int, std::vector<int>>& imgIdToLandmarkIds,
                            const std::unordered_map<int, std::vector<int>>& imgIdToFeatureIds,
                            std::vector<int>& candidateImgIdsSorted);


        /*
        Adds next image into the reconstruction via:
        1. finds image which has highest number of matches with landmarks(previously triangulated features)
        2. does solvePnp with RANSAC to estimate camera position
        3. performs global bundle adjustment
        */
        void addNextView();

        /*
            Calculates camera pose(SE(3)) via solvePnP wrapped by RANSAC
        */
        Eigen::Matrix4d registerImagePnP(const int imgIdx,
                                         std::vector<int> &featureIdxs,
                                         std::vector<int> &landmarkIdxs);

        /*
            Performs end2end reconstruction using all images in `imgFolder`
            saves results(feature matches + reconstructed clouds) in `outFolder`
        */
        void reconstruct(const std::string &imgFolder,
                         const std::string &outFolder);

        /*
            Calculate the angle between rays coming from camera centers to
            triangulated landmark
        */
        double calcTriangulationAngle(int imgIdx1, int imgIdx2,
                                      int featIdx1, int featIdx2,
                                      const Landmark &landmark);

        /*
            Transforms landmark into a given camera frame
        */
        Eigen::Vector3d getLandmarkLocalCoords(int imgIdx,
                                               Landmark landmark);

        /*
            Calculates the landmark projection error:
            [u',v',1] = K * [R|t] * X
        */
        double calcProjectionError(int imgIdx, int featIdx,
                                   Eigen::Vector3d landmarkCoordsLocal);

        /*
            Checks whether landmarks are valid or not,
            helps with outlier removal
        */
        std::vector<bool> checkLandmarkValidity();

        /*
            Removes all landmarks that were marked as outliers
        */
        void removeOutlierLandmarks(const std::vector<bool> &inlierIds);

    private:
        // stores pairs of (imgId : imgPath)
        std::unordered_map<int, fs::path> imgIds2Paths;

        // stores features per imgId,
        std::unordered_map<int, std::vector<FeaturePtr<>>> features;

        // stores triangulated features
        std::vector<Landmark> landmarks;

        // stores camera poses as extrinsics
        std::unordered_map<int, Eigen::Matrix4d> imgIdx2camPose;

        // stores camera intrinsics
        std::unordered_map<int, PinholeCamera> imgIdx2camIntrinsics;

        // stores whether corresponding imgId was registered
        std::unordered_map<int, bool> registeredImages;

        // stores images in reconstruction order
        std::vector<int> imgIdxOrder;

        // imgMatches[imgId] - contains vector of all matched image ids
        std::unordered_map<int, std::vector<int>> imgMatches;

        // stores matched feature ids per img pair
        std::unordered_map<std::pair<int, int>, std::unordered_map<int, int>, pair_hash> featureMatches;

        // stores images sizes(necessary for feature coords normalization on superglue)
        std::unordered_map<int, std::pair<int, int>> imgIdx2imgShape;

        // pointers to the underlying algorithms
        std::unique_ptr<FeatureDetector> featDetector;
        std::unique_ptr<FeatureMatcher> featMatcher;
        std::unique_ptr<ImageMatcher> imgMatcher;
        std::unique_ptr<GeometricFilter> featFilter;

        NextImageRankingMode nextImageRankingMode = NextImageRankingMode::MatchDensity;
        
        // min number of 2d-3d matches to consider image for pnp
        int min2d3dMatchNum = 30;

        // class for execution time profiling
        TimeLogger timeLogger;

        // max possible img side
        unsigned imgMaxSize = 512;

        // in case camera calibration is known, use them
        double defaultFov = 30.7;
        double defaultFocalLengthmm = 11.6;
        
        double defaultFocalLengthPxX = -1;
        double defaultFocalLengthPxY = -1;

        // variables which control whether landmark is inlier/outlier
        double maxProjectionError = 4.0;
        double minTriangulationAngle = 1.0;

        // in colmap, focal length initialized as: focalLength = focalLengthFactor * max(width, height)
        // here same initialization is used(in case intrinsics are unknown)
        double defaultFocalLengthFactor = 1.2;
        double downScaleFactor = 1.0;
    };

} // namespace reconstructor::Core