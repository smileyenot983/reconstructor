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
                                                                               reconstructor::Core::FeatMatcherType::SuperGlue,
                                                                               reconstructor::Core::ImgMatcherType::Fake);

    reconstructor->reconstruct("../data/", "../out_data/");

    return 0;
}