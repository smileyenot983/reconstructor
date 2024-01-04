#pragma once

#include <vector>

#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>

// #include "opencv2/xfeatures2d.hpp"

#include "datatypes.h"

namespace cv
{
    class Mat;
}

namespace reconstructor::Core
{
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

        // virtual void detect(const cv::Mat &img,
        //                     std::vector<FeatureConf<>> &features) = 0;

        virtual void detect(const cv::Mat &img,
                            std::vector<FeaturePtr<>> &features) = 0;

        // preprocess image resize/change data type
        virtual cv::Mat prepImg(const cv::Mat &img) = 0;

        virtual ~FeatureDetector() {}
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
                    std::vector<FeaturePtr<>> &features) override;

        cv::Mat prepImg(const cv::Mat &img) override;

    private:
        cv::Ptr<cv::ORB> orbDetector;
        cv::Ptr<cv::SIFT> siftDetector;

        int inputImgType = CV_8U;
    };

} // namespace reconstructor::Core
