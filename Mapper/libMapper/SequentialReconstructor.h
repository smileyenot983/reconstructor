#pragma once

#include <string>
#include <filesystem>

#include "FeatureDetector.h"
#include "FeatureSuperPoint.h"
#include "FeatureMatcher.h"
#include "FeatureMatcherSuperglue.h"

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
                                const FeatMatcherType& featMatcherType);

        // used for performing feature detection on all images in folder
        void detectFeatures();

        // creates pairs of images for feature matching
        void matchImages();

        // used for performing feature matching on all images in folder
        void matchFeatures();

        void filterFeatMatches();

        // chooses initial image pair to start reconstruction
        void chooseInitialPair();

        // performs end2end reconstruction
        void reconstruct(const std::string& imgFolder);
    
    private:

        // stores pairs of (imgId : imgPath) 
        std::vector<std::pair<int, fs::path>> imgIds2Paths;
        // stores features per imgId
        std::vector<std::vector<FeaturePtr<>>> features;
        // stores pairs of matched image ids
        std::vector<std::pair<int, int>> imgMatches;
        // stores map, { (imgId1, imgId2) : (matched feature ids)} 
        std::map<std::pair<int, int>, std::vector<int>> featureMatches;

        std::unique_ptr<FeatureDetector> featDetector;
        std::unique_ptr<FeatureMatcher> featMatcher;

        unsigned imgMaxSize = 512;

    };


} // namespace reconstructor::Core