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
        void drawFeaturesAndSave(const std::string& outFolder);

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
        void triangulateInitialPair(const Eigen::Matrix4d relativePose,
                                                         const int imgIdx1,
                                                         const int imgIdx2);

        void addNextView();

        // registers image via solving pnp
        void registerImagePnP(const int imgIdx,
                              const std::vector<int>& featureIdxs,
                              const std::vector<int>& landmarkIdxs);

        // geometrically filters
        void filterFeatureMatches();

        // performs end2end reconstruction
        void reconstruct(const std::string& imgFolder,
                         const std::string& outFolder);
    
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
        std::vector<std::pair<int, int>> imgShapes;

        std::unique_ptr<FeatureDetector> featDetector;
        std::unique_ptr<FeatureMatcher> featMatcher;
        std::unique_ptr<ImageMatcher> imgMatcher;
        std::unique_ptr<GeometricFilter> featFilter;

        

        unsigned imgMaxSize = 512;

        double defaultFov = 47.4;
        double defaultFocalLengthmm = 42.0;

    };


} // namespace reconstructor::Core