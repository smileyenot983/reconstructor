// TODO:
// 1. make ImgPreparator class

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>  

#include <libMapper/SequentialReconstructor.h>

int main()
{
    std::unique_ptr<reconstructor::Core::SequentialReconstructor> reconstructor = 
                std::make_unique<reconstructor::Core::SequentialReconstructor>(reconstructor::Core::FeatDetectorType::Classic,
                                                                               reconstructor::Core::FeatMatcherType::Flann,
                                                                               reconstructor::Core::ImgMatcherType::Fake);

    reconstructor->reconstruct("../data/", "../out_data");

    return 0;
}