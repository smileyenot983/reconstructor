#pragma once

#include <vector>

#include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"

namespace cv
{
    class Mat;
}

namespace reconstructor::Core
{
    struct FeatCoord
    {
        FeatCoord() {}

        FeatCoord(int x, int y)
            : x(x), y(y)
        {
        }

        int x;
        int y;
    };

    // TODO:
    // check how initialization using constructor is done;
    struct FeatDesc
    {
        FeatDesc() {}

        // the only way to call vector constructor using iterators
        template <typename InputIterator>
        FeatDesc(InputIterator first, InputIterator last)
            : desc(first, last)
        {
        }
        std::vector<float> desc;
    };

    /*
    Struct for storing feature with coords and descriptor
    */
    struct Feature
    {
        Feature(FeatCoord featCoord, FeatDesc featDesc)
            : featCoord(featCoord), featDesc(featDesc)
        {
        }
        Feature() {}
        FeatCoord featCoord;
        FeatDesc featDesc;
    };

    /*
    Base class for feature detection:
    @param[in] image - input image, for detecting features
    @param[out] feat_coords - x,y coords of features
    @param[out] feat_descs - descriptors of features
    */
    class FeatureDetector
    {
    public:
        // FeatureDetector();

        virtual void detect(const cv::Mat &img,
                            std::vector<Feature> &features) = 0;

        // preprocess image resize/change data type
        virtual cv::Mat prepImg(const cv::Mat &img) = 0;
    };

    /*
    Minimal example of feature detector based on OpenCV
    for feature detection and descriptions
    */
    // Elapsed time: 3.96186 s
    class FeatureORB : public FeatureDetector
    {
    public:
        FeatureORB();

        void detect(const cv::Mat &img,
                    std::vector<Feature> &features) override;

        cv::Mat prepImg(const cv::Mat &img) override;

    private:
        cv::Ptr<cv::ORB> orbDetector;

        int inputImgType = CV_8U;
    };

} // namespace reconstructor::Core
