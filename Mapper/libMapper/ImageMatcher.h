#pragma once

#include <filesystem>

#include "datatypes.h"

namespace fs = std::filesystem;

namespace reconstructor::Core
{
    /*
    Abstract class for image matching
    */
    class ImageMatcher
    {
    public:
        virtual void match(const std::unordered_map<int, fs::path>& imgIds2Paths,
                           const std::unordered_map<int, std::vector<FeaturePtr<>>>& features,
                           std::unordered_map<int, std::vector<int>>& imgMatches) = 0;
    
        virtual ~ImageMatcher(){};
    };

    /*
    This class creates all possible pairs of images(as a temporary solution for img matching)
    */
    class FakeImgMatcher : public ImageMatcher
    {
    public: 
        void match(const std::unordered_map<int, fs::path>& imgIds2Paths,
                   const std::unordered_map<int, std::vector<FeaturePtr<>>>& features,
                   std::unordered_map<int, std::vector<int>>& imgMatches) override;
    };

}