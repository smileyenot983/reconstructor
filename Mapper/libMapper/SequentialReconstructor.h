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

namespace fs = std::filesystem;

namespace reconstructor::Core
{
    enum FeatDetectorType
    {
        Orb,
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
        // useful for debugging to see how well feature are matched
        void drawFeatMatchesAndSave(const std::string& outFolder);

        void filterFeatMatches();

        // chooses initial image pair to start reconstruction
        Eigen::Matrix4d chooseInitialPair(int& imgIdx1, int& imgIdx2);

        // 2-view triangulation of all matched features 
        void triangulateInitialPair(const int imgIdx1,
                                    const int imgIdx2);

        // triangulate using 2+ matches
        void triangulateMultiView(const std::vector<std::pair<int, int>> matchedImgIdFeatId,
                                  bool initialCloud = false);

        void triangulateCV(const std::vector<std::pair<int, int>> matchedImgIdFeatId);

        // 
        void triangulateMatchedLandmarks(const int imgIdx,
                                         const std::vector<int>& featureIdxs,
                                         const std::vector<int>& landmarkIdxs);

        void addNextView();

        // registers image via solving pnp
        Eigen::Matrix4d registerImagePnP(const int imgIdx,
                              const std::vector<int>& featureIdxs,
                              const std::vector<int>& landmarkIdxs);

        // Eigen::Matrix4d registerImagePnpCustom(const int imgIdx,
        //                                         const std::vector<int>& featureIdxs,
        //                                         const std::vector<int>& landmarkIdxs);

        // geometrically filters
        void filterFeatureMatches();

        void adjustBundle();

        // performs end2end reconstruction
        void reconstruct(const std::string& imgFolder,
                         const std::string& outFolder);

        Eigen::Matrix3d getIntrinsics(const int imgHeight,
                           const int imgWidth);
    
    private:

        // stores pairs of (imgId : imgPath) 
        // std::vector<std::pair<int, fs::path>> imgIds2Paths;
        std::unordered_map<int, fs::path> imgIds2Paths;
        // stores features per imgId,
        // std::vector<std::vector<FeaturePtr<>>> features;

        std::unordered_map<int, std::vector<FeaturePtr<>>> features;

        std::vector<Landmark> landmarks;

        // std::vector<Eigen::Matrix4d> cameraPoses;
        // std::vector<bool> registeredImages;
        
        std::unordered_map<int, Eigen::Matrix4d> imgIdx2camPose;
        std::unordered_map<int, bool> registeredImages;

        // imgMatches[imgId] - contains vector of all matched image ids
        // std::vector<std::vector<int>> imgMatches;
        std::unordered_map< int, std::vector<int> > imgMatches;
        // stores map, { (imgId1, imgId2) : (matched feature ids, in case no matches just -1)} 
        // std::map<std::pair<int, int>, std::vector<Match>> featureMatches;

        std::unordered_map< std::pair<int, int>, std::unordered_map<int, int>, pair_hash > featureMatches;

        // stores images sizes(necessary for feature coords normalization on superglue)
        // std::vector<std::pair<int, int>> imgShapes;
        std::unordered_map<int, std::pair<int, int>> imgIdx2imgShape;

        std::unique_ptr<FeatureDetector> featDetector;
        std::unique_ptr<FeatureMatcher> featMatcher;
        std::unique_ptr<ImageMatcher> imgMatcher;
        std::unique_ptr<GeometricFilter> featFilter;

        unsigned imgMaxSize = 512;

        double defaultFov = 30.7;
        double defaultFocalLengthmm = 11.6;

        double defaultFocalLengthPx = 2759.48 / 6;

    };


} // namespace reconstructor::Core