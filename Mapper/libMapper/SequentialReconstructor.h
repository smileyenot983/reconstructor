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

    struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
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
        SequentialReconstructor(const FeatDetectorType& featDetectorType,
                                const FeatMatcherType& featMatcherType,
                                const ImgMatcherType& imgMatcherType);

        // used for performing feature detection on all images in folder
        void detectFeatures();

        // creates pairs of images for feature matching
        void matchImages();

        // used for performing feature matching on all images in folder
        void matchFeatures(bool filter = true);
        
        // draws matches between features and saves 
        void drawFeatMatchesAndSave(const std::string& outFolder);

        // filter matched features via fundamental matrix
        void filterFeatMatches();

        // chooses initial image pair to start reconstruction
        Eigen::Matrix4d chooseInitialPair(int& imgIdx1, int& imgIdx2);

        // triangulation of initial pair 
        void triangulateInitialPair(const int imgIdx1,
                                    const int imgIdx2);

        // triangulate using 2+ matches
        void triangulateMultiView(const std::vector<std::pair<int, int>> matchedImgIdFeatId,
                                  bool initialCloud = false);

        // performs linear triangulation, 1) adjusts existing landmarks
        //                                2) creates new landmarks  
        void triangulateMatchedLandmarks(const int imgIdx,
                                         const std::vector<int>& featureIdxs,
                                         const std::vector<int>& landmarkIdxs);

        void addNextView();

        // registers image via solving pnp
        Eigen::Matrix4d registerImagePnP(const int imgIdx,
                              std::vector<int>& featureIdxs,
                              std::vector<int>& landmarkIdxs);


        // performs end2end reconstruction
        void reconstruct(const std::string& imgFolder,
                         const std::string& outFolder);

        double calcTriangulationAngle(int imgIdx1, int imgIdx2,
                                      int featIdx1, int featIdx2,
                                      const Landmark& landmark);

        Eigen::Vector3d getLandmarkLocalCoords(int imgIdx,
                                               Landmark landmark);

        double calcProjectionError(int imgIdx, int featIdx,
                                    Eigen::Vector3d landmarkCoordsLocal);

        // returns whether corresponding landmarkId is valid or not
        std::vector<bool> checkLandmarkValidity();

        // removes invalid landmarks, negative depth || low triangulation angle || high reprojection error
        void removeOutlierLandmarks(const std::vector<bool>& inlierIds);


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
        std::unordered_map<int, bool> registeredImages;
        // stores images in reconstruction order
        std::vector<int> imgIdxOrder;

        // imgMatches[imgId] - contains vector of all matched image ids
        std::unordered_map< int, std::vector<int> > imgMatches;

        // stores matched feature ids per img pair
        std::unordered_map< std::pair<int, int>, std::unordered_map<int, int>, pair_hash > featureMatches;

        // stores images sizes(necessary for feature coords normalization on superglue)
        std::unordered_map<int, std::pair<int, int>> imgIdx2imgShape;

        std::unique_ptr<FeatureDetector> featDetector;
        std::unique_ptr<FeatureMatcher> featMatcher;
        std::unique_ptr<ImageMatcher> imgMatcher;
        std::unique_ptr<GeometricFilter> featFilter;

        TimeLogger timeLogger;

        unsigned imgMaxSize = 512;

        double defaultFov = 30.7; // 39.5
        double defaultFocalLengthmm = 11.6; // 50

        double maxProjectionError = 4.0;
        double minTriangulationAngle = 1.0;

        double defaultFocalLengthPxX = 1520.0 / 2;
        double defaultFocalLengthPxY = 1014 / 2;

        // in colmap, focal length initialized as: focalLength = focalLengthFactor * max(width, height)

        double defaultFocalLengthFactor = 1.2;
        double downScaleFactor = 1.0;
    };


} // namespace reconstructor::Core


// POSSIBLE IMRPOVEMENTS

// before that - run with speed profiling, to find bottlenecks

// ENGINEERING
// 1. openmp to improve speed 
// 2. think about used structures, maybe could be improved
// 3. multithreaded

// RESEARCH
// 1. improvements from sfm revisited
// 2. pixel perfect sfm
// 3. retrieve real scale using depth map estimation