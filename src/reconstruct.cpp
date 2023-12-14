// TODO:
// 1. make ImgPreparator class

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>  


#include <opencv2/opencv.hpp>

// #include <libMapper/FeatureDetector.h>
// #include <libMapper/FeatureSuperPoint.h>
// #include <libMapper/FeatureMatcher.h>
// #include <libMapper/FeatureMatcherSuperglue.h>
// #include <libMapper/utils.h>

#include <libMapper/SequentialReconstructor.h>

unsigned IMG_MAX_SIZE = 512;

class TimeLogger
{
    public:

        void startEvent(const std::string& stageName)
        {
            startTime = std::chrono::high_resolution_clock::now();
            stageNames.push_back(stageName);
            
        }
        void endEvent()
        {
            endTime = std::chrono::high_resolution_clock::now();
            auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            stageDurations.push_back(execTime.count());
        }

        void printTimings()
        {
            assert(stageNames.size() == stageDurations.size());
            std::cout << "___________Execution time profiling___________" << std::endl;
            for(size_t eventIdx = 0; eventIdx < stageDurations.size(); ++eventIdx)
            {
                std::cout << stageNames[eventIdx] << ":  " << stageDurations[eventIdx] << " ms."<< std::endl;
            }
        }

    private:

        std::vector<std::string> stageNames;
        std::vector<double> stageDurations;

        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point endTime;

};

int main()
{
    std::unique_ptr<reconstructor::Core::SequentialReconstructor> reconstructor = 
                std::make_unique<reconstructor::Core::SequentialReconstructor>(reconstructor::Core::FeatDetectorType::SuperPoint,
                                                                               reconstructor::Core::FeatMatcherType::SuperGlue);

    reconstructor->reconstruct("../data/");

    // std::unique_ptr<TimeLogger> timeLogger = std::make_unique<TimeLogger>();

    // timeLogger->startEvent("imgPrep");

    // std::string imgFolder = "../data/";

    // std::string imgPath1 = "../data/DPP_0001.JPG";
    // std::string imgPath2 = "../data/DPP_0002.JPG";

    // auto img1 = getPreparedImg(imgPath1, IMG_MAX_SIZE);
    // auto img2 = getPreparedImg(imgPath2, IMG_MAX_SIZE);
    // timeLogger->endEvent();


    // timeLogger->startEvent("featureExtraction");
    // // std::unique_ptr<reconstructor::Core::FeatureDetector> detector = std::make_unique<reconstructor::Core::FeatureORB>();
    // std::unique_ptr<reconstructor::Core::FeatureDetector> detector = std::make_unique<reconstructor::Core::FeatureSuperPoint>("../models/superpoint_model.zip");
    // auto imgPrepared1 = detector->prepImg(img1);
    // auto imgPrepared2 = detector->prepImg(img2);
    
    // // features are pointers to be able to use dynamic polymorphism
    // std::vector<reconstructor::Core::FeaturePtr<>> features1;
    // std::vector<reconstructor::Core::FeaturePtr<>> features2;
    // detector->detect(imgPrepared1, features1);
    // detector->detect(imgPrepared2, features2);

    // std::shared_ptr<reconstructor::Core::FeatureConf<>> featConf = std::dynamic_pointer_cast<reconstructor::Core::FeatureConf<>>(features1[0]);
    // std::shared_ptr<reconstructor::Core::FeatureConf<>> featConf1 = std::dynamic_pointer_cast<reconstructor::Core::FeatureConf<>>(features1[1]);
    // std::shared_ptr<reconstructor::Core::FeatureConf<>> featConf2 = std::dynamic_pointer_cast<reconstructor::Core::FeatureConf<>>(features1[2]);

    // std::cout << "n features img1: " << features1.size() << std::endl;
    // std::cout << "n features img2: " << features2.size() << std::endl;
    // // std::cout << "features1[0].featDesc.desc[0]: " << features1[0].featDesc.desc[0] << std::endl;

    // timeLogger->endEvent();

    // timeLogger->startEvent("featMatching");
    // std::unique_ptr<reconstructor::Core::FeatureMatcher> featMatcher = 
    //                         std::make_unique<reconstructor::Core::FeatureMatcherSuperglue>(imgPrepared1.rows,
    //                                                                                        imgPrepared1.cols,
    //                                                                                        "../models/superglue_model.zip");
    // // std::unique_ptr<reconstructor::Core::FeatureMatcher> featMatcher = std::make_unique<reconstructor::Core::FlannMatcher>();
    
    
    // std::vector<reconstructor::Core::Match> featMatches;

    // featMatcher->matchFeatures(features1,
    //                            features2,
    //                            featMatches);

    // std::cout << "n matches: " << featMatches.size() << std::endl;
    
    // timeLogger->endEvent();
    // timeLogger->printTimings();


    // reconstructor::Utils::visualizeKeypoints(img1, features1, 1, true);
    // reconstructor::Utils::visualizeKeypoints(img2, features2, 2, true);


    return 0;
}