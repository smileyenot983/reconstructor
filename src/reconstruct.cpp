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

#include <libMapper/SequentialReconstructor.h>


int main()
{
    std::unique_ptr<reconstructor::Core::SequentialReconstructor> reconstructor = 
                std::make_unique<reconstructor::Core::SequentialReconstructor>(reconstructor::Core::FeatDetectorType::SuperPoint,
                                                                               reconstructor::Core::FeatMatcherType::SuperGlue,
                                                                               reconstructor::Core::ImgMatcherType::Fake);

    reconstructor->reconstruct("../data/", "../out_data");

    return 0;
}