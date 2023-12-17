#pragma once

#include <string>
#include <filesystem>

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
        void matchFeatures();

        void filterFeatMatches();

        // chooses initial image pair to start reconstruction
        void chooseInitialPair();

        // geometrically filters
        void filterFeatureMatches();

        // performs end2end reconstruction
        void reconstruct(const std::string& imgFolder);
    
    private:

        // stores pairs of (imgId : imgPath) 
        std::vector<std::pair<int, fs::path>> imgIds2Paths;
        // stores features per imgId, indexed by viewId
        std::vector<std::vector<FeaturePtr<>>> features;
        // imgMatches[imgId] - contains vector of all matched image ids
        std::vector<std::vector<int>> imgMatches;
        // stores map, { (imgId1, imgId2) : (matched feature ids, in case no matches just -1)} 
        std::map<std::pair<int, int>, std::vector<Match>> featureMatches;

        // stores images sizes(necessary for feature coords normalization on superglue)
        std::vector<std::pair<int, int>> imgShapes;

        std::unique_ptr<FeatureDetector> featDetector;
        std::unique_ptr<FeatureMatcher> featMatcher;
        std::unique_ptr<ImageMatcher> imgMatcher;
        std::unique_ptr<GeometricFilter> featFilter;

        

        unsigned imgMaxSize = 512;

    };


} // namespace reconstructor::Core